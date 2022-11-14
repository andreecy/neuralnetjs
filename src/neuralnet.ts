import { promises as fs } from "fs"
// import * as nj from "numjs"
import * as math from "mathjs"

export class NeuralNet {
    inputCount: number
    hiddenCount: number
    outputCount: number
    weight0: math.MathCollection
    weight1: math.MathCollection

    constructor(inputCount, hiddenCount, outputCount) {
        this.inputCount = inputCount
        this.hiddenCount = hiddenCount
        this.outputCount = outputCount

        // matrix shape randrom value between -1 and 1
        this.weight0 = math.random([inputCount, hiddenCount], -1, 1)
        this.weight1 = math.random([hiddenCount, outputCount], -1, 1)
    }

    inspect() {
        console.log('inspecting model:', this)
    }

    // helper for sigmoid function
    sigmoid(x: math.MathCollection) {
        // 1 / (1 + exp(-x))
        let negX = math.multiply(-1, x) as math.MathCollection
        let expNegX = math.map(negX, math.exp)
        let onePlusExpNegX = math.add(1, expNegX) as math.MathCollection
        // console.log('onePlusExpNegX', onePlusExpNegX)
        return math.map(onePlusExpNegX, x => 1 / x)
    }

    // helper for calculate sigmoid dervative
    sigmoidDerivative(x: math.MathCollection) {
        // x * (1 - x)
        let oneSubX = math.subtract(1, x)
        // console.log('sigmoidDerivative')
        // console.log('x', x)
        // console.log('oneSubX', oneSubX)
        return math.dotMultiply(x, oneSubX);
    }

    // feed forward algorithm
    layerForward(inputs: math.MathCollection, weights: math.MathCollection) {
        // generate layer outputs
        // matrix dot product inputs
        // console.log('inputs', inputs)
        // console.log('weights', weights)
        let dotInputs = math.multiply(inputs, weights) as math.MathCollection
        // console.log('dotInputs', dotInputs)
        // activate function
        let outputs = this.sigmoid(dotInputs)
        return outputs as math.MathCollection
    }

    /**
     * train with input and target dataset
     * @param inputs X input dataset
     * @param targets y validation dataset, expected target output 
     */
    train(inputs: math.MathCollection, targets: math.MathCollection) {
        let hiddens = this.layerForward(inputs, this.weight0)
        // console.log(hiddens)
        let outputs = this.layerForward(hiddens, this.weight1)
        // console.log('outputs',outputs)

        // calculate output layer error
        let outputErrors = math.subtract(targets, outputs)
        // calculate Gradient descent, find best hidden-output weights
        let outputGradient = math.dotMultiply(this.sigmoidDerivative(outputs), outputErrors)
        let weightOutputDelta = math.multiply(math.transpose(hiddens), outputGradient)
        // adjust synapse weight hidden-output with delta
        this.weight1 = math.add(this.weight1, weightOutputDelta) as math.MathCollection

        // calculate hidden layer error
        let hiddenErrors = math.multiply(outputErrors, math.transpose(this.weight1))
        // calculate Gradient descent, find best input-hidden weights
        let hiddenGradient = math.dotMultiply(this.sigmoidDerivative(hiddens), hiddenErrors)
        let weightHiddenDelta = math.multiply(math.transpose(inputs), hiddenGradient)
        // adjust synapse weight input-hidden with delta
        this.weight0 = math.add(this.weight0, weightHiddenDelta) as math.MathCollection
    }

    /**
     * train with input and target dataset
     * @param inputs X input dataset
     * @param targets y validation dataset, expected target output 
     * @param epoch itteration to train
     */
    fit(inputs: math.MathCollection, targets: math.MathCollection, epoch: number = 10) {
        for (let i = 0; i < epoch; i++) {
            this.train(inputs, targets)
        }
    }

    async load(filepath: string = "model.json") {
        console.log('loading model: ', filepath);
        await fs.readFile(filepath, 'utf8')
            .then((str) => {
                const data = JSON.parse(str);
                this.inputCount = data.inputCount;
                this.hiddenCount = data.hiddenCount;
                this.outputCount = data.outputCount;
                this.weight0 = data.weight0;
                this.weight1 = data.weight1;
                console.log("Trained model has been loaded.");
                console.log(this);
            })
            .catch((err) => {
                if (err) {
                    console.log("An error occured while reading JSON Object to File.");
                    return console.log(err);
                }
            });
    }

    async save(filepath: string = "model.json") {
        console.log('saving model: ', filepath);
        const data = JSON.stringify(this)
        await fs.writeFile(filepath, data, 'utf8')
            .then(() => {
                console.log("Trained model has been saved.");
            })
            .catch((err) => {
                console.log("An error occured while writing JSON Object to File.");
                return console.log(err);
            });
    }

    /**
     * predict our trained model
     * @param inputs 
     * @returns prediciton
     */
    predict(inputs: math.MathCollection) {
        let hiddens = this.layerForward(inputs, this.weight0)
        let outputs = this.layerForward(hiddens, this.weight1)
        return outputs
    }

}
