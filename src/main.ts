import * as math from "mathjs"
import { NeuralNet } from "./neuralnet";

console.time("process")

// train dataset
const X = [
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 1],
    [1, 1, 0],
]

// validation dataset
const y = math.transpose([[
    0,
    0,
    1,
    1
]])


const model = new NeuralNet(3, 2, 1)
model.inspect()

// train our model
model.fit(X, y, 10)
await model.save()

await model.load()

// test dataset
let test = [
    [0, 1, 1]
]

let prediction = model.predict(test)
console.log('prediction: ', prediction)
console.timeEnd("process");