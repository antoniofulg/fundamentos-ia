import "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js"
import { workerEvents } from "../events/constants.js"

console.log("Model training worker initialized")
let _globalCtx = null
let _model = null

const WEIGHTS = {
	category: 0.4,
	color: 0.3,
	price: 0.2,
	age: 0.1,
}

// Normalize continuous values (price, age) to 0-1 range
// Why? Keeps all features balanced so no one dominates training
// Formula: (value - min) / (max - min)
// Example: price=129.99, minPrice=39.99, maxPrice=199,99 -> 0.56
const normalize = (value, min, max) => (value - min) / (max - min || 1)

function makeContext(products, users) {
	const ages = users.map((user) => user.age)
	const prices = products.map((product) => product.price)

	const minAge = Math.min(...ages)
	const maxAge = Math.max(...ages)

	const minPrice = Math.min(...prices)
	const maxPrice = Math.max(...prices)

	const colors = [...new Set(products.map((product) => product.color))]
	const categories = [...new Set(products.map((product) => product.category))]

	const colorsIndex = Object.fromEntries(
		colors.map((color, index) => {
			return [color, index]
		}),
	)

	const categoriesIndex = Object.fromEntries(
		categories.map((category, index) => {
			return [category, index]
		}),
	)

	// Computar a média de idade dos compradores por produto
	// (ajuda a personalizar)
	const avgAge = (minAge + maxAge) / 2
	const ageSums = {}
	const ageCounts = {}

	users.forEach((user) => {
		user.purchases.forEach((purchase) => {
			ageSums[purchase.name] = (ageSums[purchase.name] || 0) + user.age
			ageCounts[purchase.name] = (ageCounts[purchase.name] || 0) + 1
		})
	})

	const productAvgAgeNorm = Object.fromEntries(
		products.map((product) => {
			const avg = ageCounts[product.name]
				? ageSums[product.name] / ageCounts[product.name]
				: avgAge

			return [product.name, normalize(avg, minAge, maxAge)]
		}),
	)

	return {
		products,
		users,
		colorsIndex,
		categoriesIndex,
		minAge,
		maxAge,
		minPrice,
		maxPrice,
		productAvgAgeNorm,
		numCategories: categories.length,
		numColors: colors.length,
		// (price + age = 2) + colors + categories
		dimensions: 2 + categories.length + colors.length,
	}
}

const oneHotWeighted = (index, length, weight) =>
	tf.oneHot(index, length).cast("float32").mul(weight)

function encodeProduct(product, context) {
	// Normalizando dados para ficar de 0 a 1 e aplicar peso na recomendação
	const price = tf
		.tensor1d([normalize(product.price, context.minPrice, context.maxPrice)])
		.mul(WEIGHTS.price)

	const age = tf.tensor1d([
		(context.productAvgAgeNorm[product.name] ?? 0.5) * WEIGHTS.age,
	])

	const category = oneHotWeighted(
		context.categoriesIndex[product.category],
		context.numCategories,
		WEIGHTS.category,
	)

	const color = oneHotWeighted(
		context.colorsIndex[product.color],
		context.numColors,
		WEIGHTS.color,
	)

	return tf.concat([price, age, category, color], 0)
}

function encodeUser(user, context) {
	if (user.purchases.length) {
		return tf
			.stack(user.purchases.map((purchase) => encodeProduct(purchase, context)))
			.mean(0)
			.reshape([1, context.dimensions])
	}
}

function createTrainingData(context) {
	const inputs = []
	const labels = []

	context.users
		.filter((user) => user.purchases.length)
		.forEach((user) => {
			const userVector = encodeUser(user, context).dataSync()
			context.products.forEach((product) => {
				const productVector = encodeProduct(product, context).dataSync()

				const label = user.purchases.some((purchase) =>
					purchase.name === product.name ? 1 : 0,
				)
				// combinar user + product
				inputs.push([...userVector, ...productVector])
				labels.push(label)
			})
		})

	return {
		xs: tf.tensor2d(inputs),
		ys: tf.tensor2d(labels, [labels.length, 1]),
		// user vector + product vector
		inputDimension: context.dimensions * 2,
	}
}

// ====================================================================
// 📌 Exemplo de como um usuário é ANTES da codificação
// ====================================================================
/*
const exampleUser = {
    id: 201,
    name: 'Rafael Souza',
    age: 27,
    purchases: [
        { id: 8, name: 'Boné Estiloso', category: 'acessórios', price: 39.99, color: 'preto' },
        { id: 9, name: 'Mochila Executiva', category: 'acessórios', price: 159.99, color: 'cinza' }
    ]
};
*/

// ====================================================================
// 📌 Após a codificação, o modelo NÃO vê nomes ou palavras.
// Ele vê um VETOR NUMÉRICO (todos normalizados entre 0–1).
// Exemplo: [preço_normalizado, idade_normalizada, cat_one_hot..., cor_one_hot...]
//
// Suponha categorias = ['acessórios', 'eletrônicos', 'vestuário']
// Suponha cores      = ['preto', 'cinza', 'azul']
//
// Para Rafael (idade 27, categoria: acessórios, cores: preto/cinza),
// o vetor poderia ficar assim:
//
// [
//   0.45,            // peso do preço normalizado
//   0.60,            // idade normalizada
//   1, 0, 0,         // one-hot de categoria (acessórios = ativo)
//   1, 0, 0          // one-hot de cores (preto e cinza ativos, azul inativo)
// ]
//
// São esses números que vão para a rede neural.
// ====================================================================

// ====================================================================
// 🧠 Configuração e treinamento da rede neural
// ====================================================================
async function configureNeuralNetAndTrain(trainingData) {
	const model = tf.sequential()

	// Camada de entrada
	// - inputShape: Número de features por exemplo de treino (trainData.inputDim)
	//   Exemplo: Se o vetor produto + usuário = 20 números, então inputDim = 20
	// - units: 128 neurônios (muitos "olhos" para detectar padrões)
	// - activation: 'relu' (mantém apenas sinais positivos, ajuda a aprender padrões não-lineares)
	model.add(
		tf.layers.dense({
			inputShape: [trainingData.inputDimension],
			units: 128,
			activation: "relu",
		}),
	)

	// Camada oculta 1
	// - 64 neurônios (menos que a primeira camada: começa a comprimir informação)
	// - activation: 'relu' (ainda extraindo combinações relevantes de features)
	model.add(
		tf.layers.dense({
			units: 64,
			activation: "relu",
		}),
	)

	// Camada oculta 2
	// - 32 neurônios (mais estreita de novo, destilando as informações mais importantes)
	//   Exemplo: De muitos sinais, mantém apenas os padrões mais fortes
	// - activation: 'relu'
	model.add(
		tf.layers.dense({
			units: 32,
			activation: "relu",
		}),
	)

	// Camada de saída
	// - 1 neurônio porque vamos retornar apenas uma pontuação de recomendação
	// - activation: 'sigmoid' comprime o resultado para o intervalo 0–1
	//   Exemplo: 0.9 = recomendação forte, 0.1 = recomendação fraca
	model.add(
		tf.layers.dense({
			units: 1,
			activation: "sigmoid",
		}),
	)

	model.compile({
		optimizer: tf.train.adam(0.01),
		loss: "binaryCrossentropy",
		metrics: ["accuracy"],
	})

	await model.fit(trainingData.xs, trainingData.ys, {
		epochs: 100,
		batchSize: 32,
		shuffle: true,
		callbacks: {
			onEpochEnd: (epoch, logs) => {
				postMessage({
					type: workerEvents.trainingLog,
					epoch,
					loss: logs.loss,
					accuracy: logs.acc,
				})
			},
		},
	})

	return model
}

async function trainModel({ users }) {
	console.log("Training model with users:", users)

	postMessage({ type: workerEvents.progressUpdate, progress: { progress: 50 } })

	// Use root-relative URL: worker script is at /src/workers/ so ./data would be wrong
	const products = await (await fetch("/data/products.json")).json()

	const context = makeContext(products, users)
	context.productVectors = products.map((product) => {
		return {
			name: product.name,
			meta: { ...product },
			vector: encodeProduct(product, context).dataSync(),
		}
	})

	_globalCtx = context

	const trainingData = createTrainingData(context)

	_model = await configureNeuralNetAndTrain(trainingData)

	postMessage({
		type: workerEvents.progressUpdate,
		progress: { progress: 100 },
	})
	postMessage({ type: workerEvents.trainingComplete })
}

function recommend(user, ctx) {
	console.log("will recommend for user:", user)
	// postMessage({
	//     type: workerEvents.recommend,
	//     user,
	//     recommendations: []
	// });
}

const handlers = {
	[workerEvents.trainModel]: trainModel,
	[workerEvents.recommend]: (d) => recommend(d.user, _globalCtx),
}

self.onmessage = (e) => {
	const { action, ...data } = e.data
	if (handlers[action]) handlers[action](data)
}
