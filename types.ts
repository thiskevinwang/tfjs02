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
