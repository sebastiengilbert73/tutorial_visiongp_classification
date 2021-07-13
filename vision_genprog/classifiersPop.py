import genprog.evolution as gpevo
import genprog
from typing import Dict, List, Any, Set, Optional, Union, Tuple
import numpy as np


class ClassifiersPopulation(gpevo.Population):
    def __init__(self):
        pass

    def EvaluateIndividualCosts(self, inputOutputTuplesList: List[ Tuple[ Dict[str, Any], Any ] ],
                                variableNameToTypeDict: Dict[str, str],
                                interpreter: genprog.core.Interpreter,
                                returnType: str,
                                weightForNumberOfElements: float) -> Dict[genprog.core.Individual, float]:
        individual_to_cost = {}
        if len(inputOutputTuplesList) == 0:
            raise ValueError("classifiersPop.ClassifiersPopulation.EvaluateIndividualCosts(): len(inputOutputTuplesList) == 0")
        for individual in self._individualsList:
            cost_sum = 0
            for inputOutput in inputOutputTuplesList:
                variableName_to_value = inputOutput[0]
                target_class_index = inputOutput[1]
                predicted_class_vector = interpreter.Evaluate(individual, variableNameToTypeDict,
                                                             variableName_to_value, returnType)
                predicted_class_index = np.argmax(predicted_class_vector)
                if predicted_class_index != target_class_index:
                    cost_sum += 1
            individual_to_cost[individual] = cost_sum / len(inputOutputTuplesList)

        if weightForNumberOfElements != 0:
            for individual in self._individualsList:
                individual_to_cost[individual] += weightForNumberOfElements * len(individual.Elements())

        return individual_to_cost


def InputToPrediction(individual, variableNameToValue_list, interpreter, variableName_to_type,
                      return_type):
    correspondingPrediction_list = []
    for variableName_to_value in variableNameToValue_list:
        predicted_class_vector = interpreter.Evaluate(
            individual,
            variableName_to_type,
            variableName_to_value,
            return_type)
        correspondingPrediction_list.append(predicted_class_vector)
    return correspondingPrediction_list

def Accuracy(individual, inputOutput_list, interpreter, variableName_to_type,
                      return_type):
    if len(inputOutput_list) == 0:
        raise ValueError("classifiersPop.Accuracy(): Empty input-output list")
    correspondingPredictions_list = InputToPrediction(individual, [input for (input, output) in inputOutput_list],
                                            interpreter, variableName_to_type, return_type)
    number_of_correct_predictions = 0
    for sampleNdx in range(len(inputOutput_list)):
        #variableName_to_value = inputOutput_list[sampleNdx][0]
        target_classNdx = inputOutput_list[sampleNdx][1]
        prediction_vector = correspondingPredictions_list[sampleNdx]
        if np.argmax(prediction_vector) == target_classNdx:
            number_of_correct_predictions += 1
    return number_of_correct_predictions/len(inputOutput_list)
