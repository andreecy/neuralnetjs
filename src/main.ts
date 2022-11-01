// import * as nj from "numjs"
import * as math from "mathjs"
import { NeuralNet } from "./neuralnet";

console.time("process")

const model = new NeuralNet(3, 2, 1)

// train dataset
let X = [
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 1],
    [1, 1, 0],
]

// validation dataset
let y = math.transpose([[
    0,
    0,
    1,
    1
]])

// train our model
for (let i = 0; i < 20000; i++) {
    model.train(X, y)
}

// test dataset
let test = [
    [0, 1, 1]
]

let prediction = model.predict(test)
console.log('prediction: ', prediction)
console.timeEnd("process");