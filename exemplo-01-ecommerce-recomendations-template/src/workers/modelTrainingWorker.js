import "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js"
import { workerEvents } from "../events/constants.js"

console.log("Model training worker initialized")
let _globalCtx = {}

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

	context.users.forEach((user) => {
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

	debugger

	postMessage({
		type: workerEvents.trainingLog,
		epoch: 1,
		loss: 1,
		accuracy: 1,
	})

	setTimeout(() => {
		postMessage({
			type: workerEvents.progressUpdate,
			progress: { progress: 100 },
		})
		postMessage({ type: workerEvents.trainingComplete })
	}, 1000)
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
