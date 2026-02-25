import tf from "@tensorflow/tfjs-node"

async function trainModel(inputXs, outputYs) {
	const model = tf.sequential()

	// Primeira cama da da rede:
	// entrada de 7 posições (idade normalizada, 3 cores, 3 localizações)

	/* 80 neurônios = pouca base de treino;
     quanto mais neurônios, mais complexidade a rede pode aprender,
     e consequentemente, mais processamento ela vai usar
  */

	/* A ativação ReLU age como um filtro:
    É como se ela deixasse somente os dados interessantes seguirem viagem na rede;
    Se a informação chegou nesse neurônio é positiva, prossegue, se for zero ou negativa, pode jogar fora pois não serve para nada.
  */
	model.add(tf.layers.dense({ units: 80, inputShape: [7], activation: "relu" }))

	// Saída: 3 neurônios = 3 categorias (premium, medium, basic)

	/* A ativação softmax normalizada a saída em probabilidades. Ele age como um "divisor de chance":
  Ele vai dividir a chance de cada categoria, por exemplo:
  premium = 0.7, medium = 0.2, basic = 0.1
  */
	model.add(tf.layers.dense({ units: 3, activation: "softmax" }))

	/* Compilando o modelo
     optimizer: "adam" = (Adaptive Moment Estimation)
     Treinador pessoal moderno para redes neurais:
     ajusta os pesos de forma eficiente e inteligente
     aprende com histórico de erros e acertos

     loss: "categoricalCrossentropy"
     compara o que o modelo "acha" (scores de categorias)
     com a resposta certa
     A categoria premium sempre será [1, 0, 0]

     Quanto mais distante da previsão do modelo da resposta correta, maior o erro (loss)

     Exemplo clássico: classificação de imagens, recomendações e categorização de usuários,
     ou qualquer coisa em que a resposta é apenas uma das várias possíveis.
  */

	model.compile({
		optimizer: "adam",
		loss: "categoricalCrossentropy",
		metrics: ["accuracy"],
	})

	// Treinamento do modelo
	/* verbose: 0 = desabilita o log interno (e usa só callback)
     epochs: 100 = quantidade de vezes que vai rodar o dataset
     shuffle: true = embaralha os dados a cada época para evitar o viés
     callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(`Epoch ${epoch} - Loss: ${logs.loss}`)
      }
    }
  */
	await model.fit(inputXs, outputYs, {
		verbose: 0,
		epochs: 100,
		shuffle: true,
		callbacks: {
			onEpochEnd: (epoch, logs) => {
				console.log(`Epoch ${epoch} - Loss: ${logs.loss}`)
			},
		},
	})

	return model
}

// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
	[0.33, 1, 0, 0, 1, 0, 0], // Erick
	[0, 0, 1, 0, 0, 1, 0], // Ana
	[1, 0, 0, 1, 0, 0, 1], // Carlos
]

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"] // Ordem dos labels
const tensorLabels = [
	[1, 0, 0], // premium - Erick
	[0, 1, 0], // medium - Ana
	[0, 0, 1], // basic - Carlos
]

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)

// QUanto mais dados, melhor
// Assim o algoritmo consegue entender melhor os padrões complexos
const model = trainModel(inputXs, outputYs)
