import * as tf from "@tensorflow/tfjs";

export interface Car {
  Acceleration: number; // 12
  Cylinders: number; // 8
  Displacement: number; // 307
  Horsepower: number; // 130
  Miles_per_Gallon: number; // 18
  Name: string; // "chevrolet chevelle malibu"
  Origin: string; // "USA"
  Weight_in_lbs: number; // 3504
  Year: string; // "1970-01-01"
}

export interface CleanedData {
  mpg: number;
  horsepower: number;
}

export interface TensorData {
  inputs: tf.Tensor<tf.Rank>;
  labels: tf.Tensor<tf.Rank>;
  inputMax: tf.Tensor<tf.Rank>;
  inputMin: tf.Tensor<tf.Rank>;
  labelMax: tf.Tensor<tf.Rank>;
  labelMin: tf.Tensor<tf.Rank>;
}

type XYZ = [number, number, number];

interface Annotation {
  leftCheek: XYZ[];
  leftEyeLower0: XYZ[];
  leftEyeLower1: XYZ[];
  leftEyeLower2: XYZ[];
  leftEyeLower3: XYZ[];
  leftEyeUpper0: XYZ[];
  leftEyeUpper1: XYZ[];
  leftEyeUpper2: XYZ[];
  leftEyebrowLower: XYZ[];
  leftEyebrowUpper: XYZ[];
  lipsLowerInner: XYZ[];
  lipsLowerOuter: XYZ[];
  lipsUpperInner: XYZ[];
  lipsUpperOuter: XYZ[];
  midwayBetweenEyes: XYZ[];
  noseBottom: XYZ[];
  noseLeftCorner: XYZ[];
  noseRightCorner: XYZ[];
  noseTip: XYZ[];
  rightCheek: XYZ[];
  rightEyeLower0: XYZ[];
  rightEyeLower1: XYZ[];
  rightEyeLower2: XYZ[];
  rightEyeLower3: XYZ[];
  rightEyeUpper0: XYZ[];
  rightEyeUpper1: XYZ[];
  rightEyeUpper2: XYZ[];
  rightEyebrowLower: XYZ[];
  rightEyebrowUpper: XYZ[];
  silhouette: XYZ[];
}
export interface FacePrediction {
  annotations: Annotation;
  boundingBox: {
    bottomRight: XYZ;
    topLeft: XYZ;
  };
  faceInViewConfidence: number;
  mesh: XYZ[];
  scaledMesh: XYZ[];
}
