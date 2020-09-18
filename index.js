console.log("Hello TensorFlow");

/**
 * Get the car data reduced to just the variables we are interested
 * and cleaned of missing data.
 */
async function getData() {
  const carsDataReq = await fetch(
    "https://storage.googleapis.com/tfjs-tutorials/carsData.json"
  );
  const carsData = await carsDataReq.json();
  const cleaned = carsData
    .map((car) => ({
      mpg: car.Miles_per_Gallon,
      horsepower: car.Horsepower,
    }))
    .filter((car) => car.mpg != null && car.horsepower != null);

  return cleaned;
}

/**
 * Define your model architecture
 * @see https://codelabs.developers.google.com/codelabs/tfjs-training-regression/index.html#3
 */
function createModel() {
  /**
   * Instantiate a sequential model
   * @type {tf.model}
   */
  const model = tf.sequential();

  // Add a single input layer
  model.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }));

  // Add an output layer
  model.add(tf.layers.dense({ units: 1, useBias: true }));

  return model;
}

async function run() {
  // Load and plot the original input data that we are going to train on.
  const data = await getData();
  const values = data.map((d) => ({
    x: d.horsepower,
    y: d.mpg,
  }));

  tfvis.render.scatterplot(
    { name: "Horsepower v MPG" },
    { values },
    {
      xLabel: "Horsepower",
      yLabel: "MPG",
      height: 300,
    }
  );

  // More code will be added below
  // Create the model
  let model = createModel();
  tfvis.show.modelSummary({ name: "Model Summary" }, model);

  /**
   * @see https://codelabs.developers.google.com/codelabs/tfjs-training-regression/index.html#5
   */
  // Convert the data to a form we can use for training.
  const tensorData = convertToTensor(data);
  const { inputs, labels } = tensorData;

  // Train the model
  await trainModel(model, inputs, labels);
  console.log("Done Training");

  // Make some predictions using the model and compare them to the original data
  testModel(model, data, tensorData);
}

document.addEventListener("DOMContentLoaded", run);

/**
 * @see https://codelabs.developers.google.com/codelabs/tfjs-training-regression/index.html#4
 * @param {*} data
 */
function convertToTensor(data) {
  // Wrapping these calculations in a tidy will dispose any
  // intermediate tensors.

  return tf.tidy(() => {
    // Step 1. Shuffle the data
    tf.util.shuffle(data);

    // Step 2. Convert data to Tensor
    const inputs = data.map((d) => d.horsepower);
    const labels = data.map((d) => d.mpg);

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

    // Step 3. Normalize the data to the range 0 - 1 using min-max scaling
    const inputMax = inputTensor.max();
    const inputMin = inputTensor.min();
    const labelMax = labelTensor.max();
    const labelMin = labelTensor.min();

    const normalizedInputs = inputTensor
      .sub(inputMin)
      .div(inputMax.sub(inputMin));
    const normalizedLabels = labelTensor
      .sub(labelMin)
      .div(labelMax.sub(labelMin));

    return {
      inputs: normalizedInputs,
      labels: normalizedLabels,
      // Return the min/max bounds so we can use them later.
      inputMax,
      inputMin,
      labelMax,
      labelMin,
    };
  });
}

/**
 *
 * @param {*} model
 * @param {*} inputs
 * @param {*} labels
 */
async function trainModel(model, inputs, labels) {
  /**
   * Prepare the model for training...
   * We have to ‘compile' the model before we train it. To do so, we have to specify a number of very important things:
   */
  model.compile({
    /**
     * `optimizer`: This is the algorithm that is going to govern the updates to the model as it sees examples.
     * There are many optimizers available in TensorFlow.js.
     * Here we have picked the adam optimizer as it is quite effective in practice and requires no configuration.
     */
    optimizer: tf.train.adam(),
    /**
     * `loss`: this is a **function** that will tell the model how well it is doing on learning each of the batches
     * (data subsets) that it is shown.
     */
    loss: tf.losses.meanSquaredError,
    /**
     * Here we use `meanSquaredError` ("mse") to compare the predictions made by the model with the true values.
     */
    metrics: ["mse"],
  });

  /**
   * `batchSize` refers to the size of the data subsets that the model will see on each iteration of training.
   * Common batch sizes tend to be in the range 32-512.
   * There isn't really an ideal batch size for all problems and it is beyond the scope of this
   * tutorial to describe the mathematical motivations for various batch sizes.
   */
  const batchSize = 32;
  /**
   * `epochs` refers to the number of times the model is going to look at the entire dataset that you provide it.
   * Here we will take 50 iterations through the dataset.
   */
  const epochs = 50;

  return await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
      { name: "Training Performance" },
      ["loss", "mse"],
      { height: 200, callbacks: ["onEpochEnd"] }
    ),
  });
}

/**
 * @see https://codelabs.developers.google.com/codelabs/tfjs-training-regression/index.html#6
 * @param {*} model
 * @param {*} inputData
 * @param {*} normalizationData
 */
function testModel(model, inputData, normalizationData) {
  const { inputMax, inputMin, labelMin, labelMax } = normalizationData;

  // Generate predictions for a uniform range of numbers between 0 and 1;
  // We un-normalize the data by doing the inverse of the min-max scaling
  // that we did earlier.
  const [xs, preds] = tf.tidy(() => {
    // We generate 100 new ‘examples' to feed to the model.
    // Model.predict is how we feed those examples into the model.
    // Note that they need to have a similar shape ([num_examples, num_features_per_example]) as when we did training.
    const xs = tf.linspace(0, 1, 100);
    const preds = model.predict(xs.reshape([100, 1]));

    const unNormXs = xs.mul(inputMax.sub(inputMin)).add(inputMin);

    const unNormPreds = preds.mul(labelMax.sub(labelMin)).add(labelMin);

    // Un-normalize the data
    return [unNormXs.dataSync(), unNormPreds.dataSync()];
  });

  const predictedPoints = Array.from(xs).map((val, i) => {
    return { x: val, y: preds[i] };
  });

  const originalPoints = inputData.map((d) => ({
    x: d.horsepower,
    y: d.mpg,
  }));

  tfvis.render.scatterplot(
    { name: "Model Predictions vs Original Data" },
    {
      values: [originalPoints, predictedPoints],
      series: ["original", "predicted"],
    },
    {
      xLabel: "Horsepower",
      yLabel: "MPG",
      height: 300,
    }
  );
}
