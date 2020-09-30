console.log("%cTensorFlow", "color:rebeccapurple; font-size:50px");

import * as assert from "assert";
import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";

/* @ts-ignore */
import * as facemesh from "@tensorflow-models/facemesh";

import type {
  Car,
  CleanedData,
  TensorData,
  AnnotatedPredictionValues,
} from "./types";

enum Tabs {
  DATA = "Data",
  MODEL = "Model",
  TRAINING = "Training",
  RESULT = "Result",
}

/**
 * An enum of html `id`s in `static/index.html`
 */
enum HtmlIds {
  RESULT = "result",
  INPUT = "input",
  VIDEO = "video",
  CANVAS = "canvas",
}

const MODEL_PATH = "localstorage://my-model-1";

/**
 * Get the car data reduced to just the variables we are interested
 * and cleaned of missing data.
 */
async function getData() {
  console.group("getData");
  console.log("⏳ Fetching Data");
  const carsDataReq = await fetch(
    "https://storage.googleapis.com/tfjs-tutorials/carsData.json"
  );
  console.log(carsDataReq);

  const carsData: Car[] = await carsDataReq.json();

  console.log("Cleaning");
  const cleaned: CleanedData[] = carsData
    .map((car) => ({
      mpg: car.Miles_per_Gallon,
      horsepower: car.Horsepower,
    }))
    .filter((car) => car.mpg != null && car.horsepower != null);

  console.log(cleaned);
  console.groupEnd();
  return cleaned;
}

/**
 * Define your model architecture
 * @see https://codelabs.developers.google.com/codelabs/tfjs-training-regression/index.html#3
 */
async function getOrCreateModel() {
  console.group("getOrCreateModel");
  let model;

  try {
    console.log(`⏳ Loading model from ${MODEL_PATH}`);
    model = await tf.loadLayersModel(MODEL_PATH);
    console.log("✅ Model Loaded");
  } catch (err) {
    console.log("err", err);
    console.log("Creating new model");
    model = tf.sequential({
      layers: [
        tf.layers.dense({ inputShape: [1], units: 64, activation: "relu" }),
        tf.layers.dense({ units: 64, activation: "relu" }),
        tf.layers.dropout({ rate: 0.2 }),
        tf.layers.dense({ units: 1 }),
      ],
    });
  }
  console.groupEnd();
  return model;
}

/**
 * @see https://codelabs.developers.google.com/codelabs/tfjs-training-regression/index.html#4
 */
function convertToTensor(data: CleanedData[]): TensorData {
  // Wrapping these calculations in a tidy will dispose any
  // intermediate tensors.

  const tensor = tf.tidy(() => {
    // Step 1. Shuffle the data
    tf.util.shuffle(data);

    // Step 2. Convert data to Tensor
    const inputs: number[] = data.map((d) => d.horsepower);
    const labels: number[] = data.map((d) => d.mpg);

    const inputTensor: tf.Tensor2D = tf.tensor2d(inputs, [inputs.length, 1]);
    const labelTensor: tf.Tensor2D = tf.tensor2d(labels, [labels.length, 1]);

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

  return tensor;
}

async function trainModel(
  model: tf.LayersModel,
  inputs: tf.Tensor<tf.Rank>,
  labels: tf.Tensor<tf.Rank>
) {
  console.group("trainModel");
  console.log("⏳ Start Training");
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
    // optimizer: tf.train.sgd(0.01),
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
  const batchSize = 100;
  /**
   * `epochs` refers to the number of times the model is going to look at the entire dataset that you provide it.
   * Here we will take 50 iterations through the dataset.
   */
  const epochs = 60;

  const result = await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
      { name: "Training Performance", tab: Tabs.TRAINING },
      ["loss", "mse"],
      { height: 200, callbacks: ["onEpochEnd"] }
    ),
  });

  console.log("✅ Done Training");
  console.groupEnd();

  return result;
}

/**
 * Make a prediction on a single user input
 */
function testUserInput(
  model: tf.LayersModel,
  /** Example: user inputs a car's horsepower. MPG will be predicted...*/
  userInput: number, // 200
  { inputMax, inputMin, labelMin, labelMax }: TensorData
) {
  console.group("testUserInput");
  console.log(
    `%cUser entered: %c${userInput} HP`,
    "font-weight:bold; font-size:16px",
    "color:lightblue; font-size:20px"
  );

  const incomingValue = tf.sub(userInput, inputMin).div(inputMax.sub(inputMin));
  console.log(`...as a tensor, ${userInput} is`);
  incomingValue.print(true);

  // console.log("...for reference...");
  // console.table([
  //   { HP_normalized: 0.33, HP_actual: 106.72 },
  //   { HP_normalized: 0.46399998664855957, HP_actual: 131.37600708007812 },
  //   { HP_normalized: 0.89, HP_actual: 209.75 },
  //   { HP_normalized: 1.0, HP_actual: 230 },
  // ]);

  const xs_1 = incomingValue.as1D();

  console.log(
    `%c${userInput} HP %cnormalized/.as1D() = `,
    "color:lightblue; font-size:20px",
    "font-weight:bold; font-size:16px"
  );
  xs_1.print();
  const x_1 = xs_1.reshape([1, 1]);
  const pred1 = model.predict(x_1) as tf.Tensor<tf.Rank>;

  // Math...
  const unNormXs_1 = xs_1.mul(inputMax.sub(inputMin)).add(inputMin);
  const unNormPred_1 = pred1.mul(labelMax.sub(labelMin)).add(labelMin);

  const horsepower = unNormXs_1.dataSync()[0];
  const milesPerGallon = unNormPred_1.dataSync()[0];
  console.table({
    HP_user_input: userInput,
    HP_normalized: x_1.dataSync()[0],
    HP_unNormalized: horsepower,
    MPG_prediction: milesPerGallon,
  });

  console.groupEnd();

  return milesPerGallon;
}

/**
 * @see https://codelabs.developers.google.com/codelabs/tfjs-training-regression/index.html#6
 */
function testModel(
  model: tf.LayersModel,
  inputData: CleanedData[],
  { inputMax, inputMin, labelMin, labelMax }: TensorData
) {
  console.group("testModel");
  console.log("⏳ Start Testing");

  // Generate predictions for a uniform range of numbers between 0 and 1;
  // We un-normalize the data by doing the inverse of the min-max scaling
  // that we did earlier.
  const [xs, preds] = tf.tidy(() => {
    // We generate 100 new ‘examples' to feed to the model.
    const xs = tf.linspace(0, 1, 100);
    const x = xs.reshape([100, 1]); // this looks like a courtesy/redundancy layer

    // Model.predict is how we feed those examples into the model.
    // Note that they need to have a similar shape ([num_examples, num_features_per_example]) as when we did training.
    const preds = model.predict(x) as tf.Tensor<tf.Rank>;

    // convert [0,0.5,...1,etc] => [40,65,...200,etc]
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
    {
      name: "Model Predictions vs Original Data",
      tab: Tabs.RESULT,
    } as tfvis.Drawable,
    {
      values: [originalPoints, predictedPoints],
      series: ["original", "predicted"],
    } as tfvis.XYPlotData,
    {
      xLabel: "Horsepower",
      yLabel: "MPG",
      height: 300,
    } as tfvis.XYPlotOptions
  );

  console.log("✅ Done Testing");
  console.groupEnd();
}

/**
 * # Primary function
 */
async function run() {
  // Load and plot the original input data that we are going to train on.
  const data = await getData();
  const values = data.map((d) => ({
    x: d.horsepower,
    y: d.mpg,
  }));

  tfvis.render.scatterplot(
    { name: "Horsepower v MPG", tab: Tabs.DATA } as tfvis.Drawable,
    { values } as tfvis.XYPlotData,
    {
      xLabel: "Horsepower",
      yLabel: "MPG",
      height: 300,
    } as tfvis.XYPlotOptions
  );

  const model = await getOrCreateModel();
  tfvis.show.modelSummary({ name: "Model Summary", tab: Tabs.DATA }, model);

  /**
   * @see https://codelabs.developers.google.com/codelabs/tfjs-training-regression/index.html#5
   */
  // Convert the data to a form we can use for training.
  const tensorData = convertToTensor(data);
  const { inputs, labels } = tensorData;

  /**
   * @see https://js.tensorflow.org/api_vis/latest/#show.valuesDistribution
   */
  await tfvis.show.valuesDistribution(
    { name: "Inputs", tab: Tabs.DATA },
    inputs
  );

  // await trainModel(model, inputs, labels);

  // Make some predictions using the model and compare them to the original data
  testModel(model, data, tensorData);
  // Make a prediction on a single user input

  function handleTestUserInput(this: HTMLInputElement, ev: Event) {
    const milesPerGallon = testUserInput(
      model,
      parseInt(this.value, 10),
      tensorData
    );
    document.getElementById(HtmlIds.RESULT).innerHTML =
      milesPerGallon.toString() + " MPG";
  }

  {
    document
      .getElementById(HtmlIds.INPUT)
      .addEventListener("change", handleTestUserInput);
  }

  console.log(`Saving model to ${MODEL_PATH}`);
  const saveResult = await model.save(MODEL_PATH);
  console.log("✅ Model saved", saveResult);
}

document.addEventListener("DOMContentLoaded", run);

async function main() {
  console.group("face detect");
  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    await navigator.mediaDevices
      .getUserMedia({
        audio: false,
        video: {
          facingMode: "user",
        },
      })
      .then((stream) => {
        console.log("camera is present");
        (window as any).stream = stream;
        const video = document.getElementById(HtmlIds.VIDEO);
        (video as any).srcObject = stream;

        return new Promise((resolve, reject) => {
          video.onloadedmetadata = () => {
            resolve(true);
          };
        });
      });
  }
  // Load the MediaPipe facemesh model.
  console.log("facemesh.load()");
  const model = await facemesh.load();
  console.log("facemesh.load() done");

  const videoElement = document.getElementById(
    HtmlIds.VIDEO
  ) as HTMLVideoElement;

  const image: tf.Tensor4D = tf.tidy(() => {
    const tensor = tf.browser.fromPixels(videoElement);
    return tensor.toFloat().expandDims(0);
  });

  // assert.deepStrictEqual(
  //   image,
  //   {
  //     dataId: {},
  //     dtype: "float32",
  //     id: 302,
  //     isDisposedInternal: false,
  //     kept: false,
  //     rankType: "4",
  //     scopeId: 214,
  //     shape: [1, 480, 640, 3],
  //     size: 921600,
  //     strides: [921600, 1920, 3],
  //     isDisposed: false,
  //     rank: 4,
  //   },
  //   "image is not equal to ___"
  // );
  console.log({ image });

  /**
   * This  will get called recursively
   */
  const renderPredictions = async () => {
    // Pass in a video stream (or an image, canvas, or 3D tensor) to obtain an
    // array of detected faces from the MediaPipe graph.
    console.log("model.estimateFaces()");
    /**
     * # estimateFaces
     * @see https://github.com/tensorflow/tfjs-models/blob/master/facemesh/src/index.ts#L204-L301
     */
    const predictions: AnnotatedPredictionValues[] = await model.estimateFaces(
      videoElement,
      true
    );
    console.log("predictions", predictions);

    const canvasElement = document.getElementById(
      HtmlIds.CANVAS
    ) as HTMLCanvasElement;
    const ctx = canvasElement.getContext("2d");
    console.log({ ctx });

    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    // Font options.
    const font = "16px sans-serif";
    ctx.font = font;
    ctx.textBaseline = "top";

    predictions.forEach((prediction, i) => {
      console.log("drawing something for prediction", i);
      const x = prediction.boundingBox.topLeft[0];
      const y = prediction.boundingBox.topLeft[1];
      const width = prediction.boundingBox.bottomRight[0] - x;
      const height = prediction.boundingBox.bottomRight[1] - y;
      console.log({ x, y, width, height });

      // Draw the bounding box.
      ctx.strokeStyle = "#00FFFF";
      ctx.lineWidth = 4;
      ctx.strokeRect(x, y, width, height);

      // Draw the label background.
      ctx.fillStyle = "#00FFFF";
      const textWidth = ctx.measureText(`Person #${i}`).width;
      const textHeight = parseInt(font, 10); // base 10
      ctx.fillRect(x, y, textWidth + 4, textHeight + 4);
    });

    predictions.forEach((prediction, i) => {
      const x = prediction.boundingBox.topLeft[0];
      const y = prediction.boundingBox.topLeft[1];
      // Draw the text last to ensure it's on top.
      ctx.fillStyle = "#000000";
      ctx.fillText(`Person #${i}`, x, y);
    });

    for (let i = 0; i < predictions.length; i++) {
      const keypoints = predictions[i].scaledMesh;

      // Log facial keypoints.
      for (let i = 0; i < keypoints.length; i++) {
        const [x, y, z] = keypoints[i];

        ctx.fillStyle = "#990099";
        ctx.fillRect(x, y, 3, 3);
        console.log(`Keypoint ${i}: [${x}, ${y}, ${z}]`);
      }
    }
  };

  const detectFrame = (video: HTMLVideoElement, model: any) => {
    renderPredictions();
    requestAnimationFrame(() => {
      detectFrame(video, model);
    });
  };
  // detectFrame(videoElement, model);
  renderPredictions();

  console.groupEnd();
}

document.addEventListener("DOMContentLoaded", main);
