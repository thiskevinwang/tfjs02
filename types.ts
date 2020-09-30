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

export type Coord2D = [number, number];
export type Coord3D = [number, number, number];
export type Coords3D = Coord3D[];

export interface AnnotatedPredictionValues {
  /** Probability of the face detection. */
  faceInViewConfidence: number;
  boundingBox: {
    /** The upper left-hand corner of the face. */
    topLeft: Coord2D;
    /** The lower right-hand corner of the face. */
    bottomRight: Coord2D;
  };
  /** Facial landmark coordinates. */
  mesh: Coords3D;
  /** Facial landmark coordinates normalized to input dimensions. */
  scaledMesh: Coords3D;
  /** Annotated keypoints. */
  annotations?: MeshAnnotations;
}

interface MeshAnnotations {
  leftCheek: number[];
  leftEyeLower0: number[];
  leftEyeLower1: number[];
  leftEyeLower2: number[];
  leftEyeLower3: number[];
  leftEyeUpper0: number[];
  leftEyeUpper1: number[];
  leftEyeUpper2: number[];
  leftEyebrowLower: number[];
  leftEyebrowUpper: number[];
  lipsLowerInner: number[];
  lipsLowerOuter: number[];
  lipsUpperInner: number[];
  lipsUpperOuter: number[];
  midwayBetweenEyes: number[];
  noseBottom: number[];
  noseLeftCorner: number[];
  noseRightCorner: number[];
  noseTip: number[];
  rightCheek: number[];
  rightEyeLower0: number[];
  rightEyeLower1: number[];
  rightEyeLower2: number[];
  rightEyeLower3: number[];
  rightEyeUpper0: number[];
  rightEyeUpper1: number[];
  rightEyeUpper2: number[];
  rightEyebrowLower: number[];
  rightEyebrowUpper: number[];
  silhouette: number[];
}
