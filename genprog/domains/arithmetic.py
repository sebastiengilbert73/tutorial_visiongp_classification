import genprog.core as gp
import genprog.evolution as gpevo
import math
from typing import Dict, List, Any, Set, Tuple, Optional, Union
import random
import pandas

possibleTypes = ['float', 'bool']
class ArithmeticInterpreter(gp.Interpreter): # An example to follow for other domains

    def FunctionDefinition(self, functionName: str, argumentsList: List[ Union[float, bool] ]) -> Union[float, bool]:
        if functionName == "addition_float":
            floatArg1: float = float(argumentsList[0])
            floatArg2: float = float(argumentsList[1])
            return floatArg1 + floatArg2
        elif functionName == "subtraction_float":
            floatArg1 = float(argumentsList[0])
            floatArg2 = float(argumentsList[1])
            return floatArg1 - floatArg2
        elif functionName == "multiplication_float":
            floatArg1 = float(argumentsList[0])
            floatArg2 = float(argumentsList[1])
            return floatArg1 * floatArg2
        elif functionName == "division_float":
            floatArg1 = float(argumentsList[0])
            floatArg2 = float(argumentsList[1])
            if floatArg2 == 0:
                return 0.0
            return floatArg1 / floatArg2
        elif functionName == "greaterThan_float":
            floatArg1 = float(argumentsList[0])
            floatArg2 = float(argumentsList[1])
            return floatArg1 > floatArg2
        elif functionName == "greaterThanOrEqual_float":
            floatArg1 = float(argumentsList[0])
            floatArg2 = float(argumentsList[1])
            return floatArg1 >= floatArg2
        elif functionName == "lessThan_float":
            floatArg1 = float(argumentsList[0])
            floatArg2 = float(argumentsList[1])
            return floatArg1 < floatArg2
        elif functionName == "lessThanOrEqual_float":
            floatArg1 = float(argumentsList[0])
            floatArg2 = float(argumentsList[1])
            return floatArg1 <= floatArg2
        elif functionName == "almostEqual_float":
            floatArg1 = float(argumentsList[0])
            floatArg2 = float(argumentsList[1])
            floatArg3: float = abs( float(argumentsList[2]) )
            return abs(floatArg1 - floatArg2) <= floatArg3
        elif functionName == "inverse_bool":
            boolArg1: bool = bool(argumentsList[0])
            return not boolArg1
        elif functionName == "log":
            floatArg1 = float(argumentsList[0])
            if floatArg1 <= 0.0:
                return 0.0
            return math.log(floatArg1)
        elif functionName == "exp":
            floatArg1 = float(argumentsList[0])
            if floatArg1 >= 20.0:
                return 0.0
            return math.exp(floatArg1)
        elif functionName == "pow_float":
            floatArg1 = float(argumentsList[0])
            floatArg2 = float(argumentsList[1])
            try:
                result: float = math.pow(floatArg1, floatArg2)
                return result
            except:
                return 0.0
        elif functionName == 'if_float':
            boolArg1 = bool(argumentsList[0])
            floatArg1 = float(argumentsList[1])
            floatArg2 = float(argumentsList[2])
            if boolArg1:
                return floatArg1
            else:
                return floatArg2
        elif functionName == 'sin':
            try:
                floatArg1 = float(argumentsList[0])
                return math.sin(floatArg1)
            except:
                return 0.0
        elif functionName == 'cos':
            try:
                floatArg1 = float(argumentsList[0])
                return math.cos(floatArg1)
            except:
                return 0.0
        elif functionName == 'tan':
            try:
                floatArg1 = float(argumentsList[0])
                return math.tan(floatArg1)
            except:
                return 0.0
        elif functionName == 'atan':
            floatArg1 = float(argumentsList[0])
            return math.atan(floatArg1)
        elif functionName == 'sigmoid':
            floatArg1 = float(argumentsList[0])
            try:
                return 1.0 / (1.0 + math.exp(-floatArg1))
            except:
                return 0.0
        elif functionName == 'ispositive_float':
            floatArg1 = float(argumentsList[0])
            return floatArg1 >= 0.0
        elif functionName == 'inverse_float':
            floatArg1 = float(argumentsList[0])
            try:
                if floatArg1 == 0:
                    return 0
                else:
                    return 1.0/floatArg1
            except:
                return 0.0
        elif functionName == 'isinbetween_float':
            floatArg1 = float(argumentsList[0])
            floatArg2 = float(argumentsList[1])
            floatArg3 = float(argumentsList[2])
            if floatArg1 >= floatArg2 and floatArg1 <= floatArg3:
                return True
            else:
                return False
        elif functionName == 'abs_float':
            floatArg1 = float(argumentsList[0])
            return abs(floatArg1)
        elif functionName == 'relu_float':
            floatArg1 = float(argumentsList[0])
            if floatArg1 >= 0:
                return floatArg1
            else:
                return 0.0
        elif functionName == 'sign_float':
            floatArg1 = float(argumentsList[0])
            if floatArg1 < 0:
                return -1.0
            elif floatArg1 > 0:
                return 1.0
            else:
                return 0.0
        elif functionName == 'max_float':
            floatArg1 = float(argumentsList[0])
            floatArg2 = float(argumentsList[1])
            return max(floatArg1, floatArg2)
        elif functionName == 'min_float':
            floatArg1 = float(argumentsList[0])
            floatArg2 = float(argumentsList[1])
            return min(floatArg1, floatArg2)
        elif functionName == 'gaussian':
            floatArg1 = float(argumentsList[0])
            floatArg2 = float(argumentsList[1])
            try:
                sigma2 = floatArg2 ** 2
                f = math.exp(-(floatArg1 ** 2)/(2 * sigma2) )
                return f
            except:
                return 0.0
        elif functionName == 'gaussian_scaled':
            x = float(argumentsList[0])
            mu = float(argumentsList[1])
            sigma = float(argumentsList[2])
            a = float(argumentsList[3])
            try:
                sigma2 = sigma**2
                return a * math.exp( -((x - mu)**2)/(2 * sigma2) )
            except:
                return 0.0
        elif functionName == 'pow2_float':
            floatArg1 = float(argumentsList[0])
            try:
                return floatArg1 ** 2
            except:
                return 0.0
        elif functionName == 'sqrt':
            floatArg1 = float(argumentsList[0])
            if floatArg1 >= 0:
                return math.sqrt(floatArg1)
            else:
                return 0.0
        elif functionName == 'windowed_linear':
            x = float(argumentsList[0]) # x
            alpha = float(argumentsList[1]) # alpha
            c = float(argumentsList[2]) # c
            L = float(argumentsList[3]) # L
            if x >= c - abs(L/2) and x <= c + abs(L/2):
                return alpha * (x - c)
            else:
                return 0
        elif functionName == 'window':
            x = float(argumentsList[0])
            alpha = float(argumentsList[1])
            c = float(argumentsList[2])
            L = float(argumentsList[3])
            if x >= c - abs(L / 2) and x <= c + abs(L / 2):
                return alpha
            else:
                return 0
        elif functionName == 'ramp_increasing':
            x = float(argumentsList[0])
            alpha = float(argumentsList[1])
            c = float(argumentsList[2])
            L = float(argumentsList[3])
            if x >= c - abs(L / 2) and x <= c + abs(L / 2):
                return alpha * (x - c + abs(L/2))
            else:
                return 0
        elif functionName == 'ramp_decreasing':
            x = float(argumentsList[0])
            alpha = float(argumentsList[1])
            c = float(argumentsList[2])
            L = float(argumentsList[3])
            if x >= c - abs(L / 2) and x <= c + abs(L / 2):
                return -alpha * (x - c - abs(L/2))
            else:
                return 0

        else:
            raise NotImplementedError("ArithmeticInterpreter.FunctionDefinition(): Not implemented function '{}'".format(functionName))

    def FunctionDerivative(self, functionName: str, argumentsList: List[ Union[float, bool] ]) -> List[ Union[float, bool] ]:
        if functionName == "addition_float":
            floatArg1: float = float(argumentsList[0])
            floatArg2: float = float(argumentsList[1])
            return [1.0, 1.0]
        elif functionName == "subtraction_float":
            floatArg1 = float(argumentsList[0])
            floatArg2 = float(argumentsList[1])
            return [1.0, -1.0]
        elif functionName == "multiplication_float":
            floatArg1 = float(argumentsList[0])
            floatArg2 = float(argumentsList[1])
            return [floatArg2 , floatArg1]
        elif functionName == "division_float":
            floatArg1 = float(argumentsList[0])
            floatArg2 = float(argumentsList[1])
            if floatArg2 == 0:
                return [0.0, 0.0]
            try:
                return [1.0/floatArg2, -1.0 * floatArg1/(floatArg2**2)]
            except:
                return [0.0, 0.0]
        elif functionName == "greaterThan_float":
            floatArg1 = float(argumentsList[0])
            floatArg2 = float(argumentsList[1])
            return [0.0 , 0.0]
        elif functionName == "greaterThanOrEqual_float":
            floatArg1 = float(argumentsList[0])
            floatArg2 = float(argumentsList[1])
            return [0.0, 0.0]
        elif functionName == "lessThan_float":
            floatArg1 = float(argumentsList[0])
            floatArg2 = float(argumentsList[1])
            return [0.0, 0.0]
        elif functionName == "lessThanOrEqual_float":
            floatArg1 = float(argumentsList[0])
            floatArg2 = float(argumentsList[1])
            return [0.0, 0.0]
        elif functionName == "almostEqual_float":
            floatArg1 = float(argumentsList[0])
            floatArg2 = float(argumentsList[1])
            floatArg3: float = abs(float(argumentsList[2]))
            return [0.0, 0.0, 0.0]
        elif functionName == "inverse_bool":
            boolArg1: bool = bool(argumentsList[0])
            return [0.0]
        elif functionName == "log":
            floatArg1 = float(argumentsList[0])
            if floatArg1 == 0.0:
                return [0.0]
            return [1.0/floatArg1]
        elif functionName == "exp":
            floatArg1 = float(argumentsList[0])
            if floatArg1 >= 20.0:
                return [0.0]
            return [math.exp(floatArg1)]
        elif functionName == "pow_float":
            floatArg1 = float(argumentsList[0])
            floatArg2 = float(argumentsList[1])
            try:
                df_dx1: float = floatArg2 * math.pow(floatArg1, floatArg2 - 1)
                df_dx2: float = math.pow(floatArg1, floatArg2) * math.log(floatArg1)
                return [df_dx1, df_dx2]
            except:
                return [0.0, 0.0]
        elif functionName == 'if_float':
            boolArg1 = bool(argumentsList[0])
            floatArg1 = float(argumentsList[1])
            floatArg2 = float(argumentsList[2])
            if boolArg1:
                return [0.0, 1.0, 0.0]
            else:
                return [0.0, 0.0, 1.0]
        elif functionName == 'sin':
            try:
                floatArg1 = float(argumentsList[0])
                return [math.cos(floatArg1)]
            except:
                return [0.0]
        elif functionName == 'cos':
            try:
                floatArg1 = float(argumentsList[0])
                return [-math.sin(floatArg1)]
            except:
                return [0.0]
        elif functionName == 'tan':
            try:
                floatArg1 = float(argumentsList[0])
                return [1.0/(math.cos(floatArg1)**2)]
            except:
                return [0.0]
        elif functionName == 'atan':
            floatArg1 = float(argumentsList[0])
            try:
                return [1.0/(1.0 + floatArg1**2)]
            except:
                return [0.0]
        elif functionName == 'sigmoid':
            floatArg1 = float(argumentsList[0])
            try:
                return [ math.exp(-floatArg1) / ( (1.0 + math.exp(-floatArg1))**2 ) ]
            except:
                return [0.0]
        elif functionName == 'ispositive_float':
            floatArg1 = float(argumentsList[0])
            return [0.0]
        elif functionName == 'inverse_float':
            floatArg1 = float(argumentsList[0])
            try:
                if floatArg1 == 0:
                    return [0.0]
                else:
                    return [-1.0 / (floatArg1**2)]
            except:
                return [0.0]
        elif functionName == 'isinbetween_float':
            floatArg1 = float(argumentsList[0])
            floatArg2 = float(argumentsList[1])
            floatArg3 = float(argumentsList[2])
            return [0.0, 0.0, 0.0]
        elif functionName == 'abs_float':
            floatArg1 = float(argumentsList[0])
            if floatArg1 >= 0:
                return [1.0]
            else:
                return [-1.0]
        elif functionName == 'relu_float':
            floatArg1 = float(argumentsList[0])
            if floatArg1 >= 0:
                return [1.0]
            else:
                return [0.0]
        elif functionName == 'sign_float':
            floatArg1 = float(argumentsList[0])
            return [0.0]
        elif functionName == 'max_float':
            floatArg1 = float(argumentsList[0])
            floatArg2 = float(argumentsList[1])
            if floatArg1 >= floatArg2:
                return [1.0, 0.0]
            else:
                return [0.0, 1.0]
        elif functionName == 'min_float':
            floatArg1 = float(argumentsList[0])
            floatArg2 = float(argumentsList[1])
            if floatArg1 <= floatArg2:
                return [1.0, 0.0]
            else:
                return [0.0, 1.0]
        elif functionName == 'gaussian':
            floatArg1 = float(argumentsList[0])
            floatArg2 = float(argumentsList[1])
            try:
                sigma2 = floatArg2 ** 2
                df_dx1 = -math.exp(-(floatArg1 ** 2) / (2 * sigma2)) * floatArg1/sigma2
                df_dx2 = math.exp(-(floatArg1 ** 2) / (2 * sigma2)) * floatArg1**2/math.pow(floatArg2, 3.0)
                return [df_dx1, df_dx2]
            except:
                return [0.0, 0.0]
        elif functionName == 'gaussian_scaled':
            x = float(argumentsList[0])
            mu = float(argumentsList[1])
            sigma = float(argumentsList[2])
            a = float(argumentsList[3])
            try:
                sigma2 = sigma**2
                df_dx = -a * (x - mu)/sigma2 * math.exp( (-(x - mu)**2)/(2 * sigma2) )
                df_dmu = a * (x - mu)/sigma2 * math.exp( (-(x - mu)**2)/(2 * sigma2) )
                df_dsigma = a * ((x - mu)**2)/(math.pow(sigma, 3)) * math.exp( (-(x - mu)**2)/(2 * sigma2) )
                df_da = math.exp( (-(x - mu)**2)/(2 * sigma2) )
                return [df_dx, df_dmu, df_dsigma, df_da]
            except:
                return [0.0, 0.0, 0.0, 0.0]
        elif functionName == 'pow2_float':
            floatArg1 = float(argumentsList[0])
            return [2.0 * floatArg1]
        elif functionName == 'sqrt':
            floatArg1 = float(argumentsList[0])
            try:
                return [0.5 * math.pow(floatArg1, -0.5)]
            except:
                return [0.0]
        elif functionName == 'windowed_linear':
            x = float(argumentsList[0]) # x
            alpha = float(argumentsList[1]) # alpha
            c = float(argumentsList[2]) # c
            L = float(argumentsList[3]) # L
            if x >= c - abs(L/2) and x <= c + abs(L/2):
                return [alpha, x - c, -alpha, 0.0]
            else:
                return [0, 0, 0, 0]
        elif functionName == 'window':
            x = float(argumentsList[0])
            alpha = float(argumentsList[1])
            c = float(argumentsList[2])
            L = float(argumentsList[3])
            if x >= c - abs(L / 2) and x <= c + abs(L / 2):
                return [0, 1, 0, 0]
            else:
                return [0, 0, 0, 0]
        elif functionName == 'ramp_increasing':
            x = float(argumentsList[0])
            alpha = float(argumentsList[1])
            c = float(argumentsList[2])
            L = float(argumentsList[3])
            if x >= c - abs(L / 2) and x <= c + abs(L / 2):
                return [alpha, x - c + abs(L)/2, -alpha, alpha/2 * numpy.sign(L)]
            else:
                return [0, 0, 0, 0]
        elif functionName == 'ramp_decreasing':
            x = float(argumentsList[0])
            alpha = float(argumentsList[1])
            c = float(argumentsList[2])
            L = float(argumentsList[3])
            if x >= c - abs(L / 2) and x <= c + abs(L / 2):
                return [-alpha, -(x - c - abs(L)/2), alpha, alpha/2 * numpy.sign(L)]
            else:
                return [0, 0, 0, 0]
        else:
            raise NotImplementedError("ArithmeticInterpreter.FunctionDerivative(): Not implemented function '{}'".format(functionName))


    def CreateConstant(self, returnType: str, parametersList: Optional[List[Union[float, bool] ]]) -> str:
        if returnType == 'float':
            if parametersList is None:
                raise ValueError("ArithmeticInterpreter.CreateConstant(): returnType = float; There is no list of parameters")
            if len(parametersList) != 2:
                raise ValueError("ArithmeticInterpreter.CreateConstant(): returnType = float; The length of the list of parameters ({}) is not 2".format(
                    len(parametersList) ))
            minValue = float(parametersList[0])
            maxValue = float(parametersList[1])
            return str( minValue + (maxValue - minValue) * random.random() )
        elif returnType == 'bool':
            if random.random() < 0.5:
                return 'True'
            else:
                return 'False'
        else:
            raise NotImplementedError("ArithmeticInterpreter.CreateConstant(): Not implemented return type '{}'".format(returnType))

    def PossibleTypes(self) -> List[str]:
        return possibleTypes



class ArithmeticPopulation(gpevo.Population): # An example to follow for other domains
    def EvaluateIndividualCosts(self,
                                inputOutputTuplesList: List[Tuple[Dict[str, Union[float, bool] ], Union[float, bool]]],
                                variableNameToTypeDict: Dict[str, str],
                                interpreter: gp.Interpreter,
                                returnType: str,
                                weightForNumberOfElements: float=0.001) -> Dict[gp.Individual, float]:
        penaltyForNoVariation = 1.0e6
        penaltyForOverflowError = 1.0e9
        individualToCostDict = {}
        if len(inputOutputTuplesList) == 0:
            raise ValueError("ArithmeticPopulation.EvaluateIndividualCosts(): The list of input-output tuples is empty")
        for individual in self._individualsList:
            costSum: float = 0.0
            anOverflowErrorOccurred: bool = False

            individualOutputsList = []
            for inputOutputPair in inputOutputTuplesList:
                input: Dict[str, Union[float, bool]] = inputOutputPair[0]
                targetOutput: Union[float, bool] = inputOutputPair[1]
                individualOutput = interpreter.Evaluate(
                    individual,
                    variableNameToTypeDict,
                    input,
                    returnType
                )
                individualOutputsList.append(individualOutput)
                if returnType == 'bool':
                    if individualOutput != targetOutput:
                        costSum += 1.0
                elif returnType == 'float':

                    try:
                        costSum += abs(individualOutput - targetOutput)# ** 2
                    except OverflowError as error:
                        #logging.warning ("OverflowError: costSum = {}; individualOutput = {}; targetOutput = {}".format(
                        #    costSum, individualOutput, targetOutput))
                        anOverflowErrorOccurred = True
                else:
                    raise NotImplementedError("ArithmeticPopulation.EvaluateIndividualCosts(): Not implemented return type '{}'".format(returnType))
            # Check if there is variation in the outputs
            aVariationWasDetected: bool = False
            for outputNdx in range(1, len(individualOutputsList)):
                if not aVariationWasDetected and individualOutputsList[outputNdx] != individualOutputsList[outputNdx - 1]:
                    aVariationWasDetected = True

            if anOverflowErrorOccurred:
                individualToCostDict[individual] = penaltyForOverflowError
            else:
                individualToCostDict[individual] = costSum / len(inputOutputTuplesList)
                if not aVariationWasDetected:
                    individualToCostDict[individual] = individualToCostDict[individual] + penaltyForNoVariation

                # Add penalty for number of elements
                numberOfElements = gpevo.NumberOfElements(individual._tree)
                individualToCostDict[individual] = individualToCostDict[individual] + weightForNumberOfElements * numberOfElements

        return individualToCostDict


def load_dataset(filepath: str, variableNameToTypeDict: Dict[str, str], outputName:str, outputType: str) -> List[Tuple[Dict[str, Union[float, bool] ], Union[float, bool]]]:
    dataframe: pandas.core.frame.DataFrame = pandas.read_csv(filepath)
    # Check that variable names and output name appear in the columns, variable types and output type are possible types
    for variableName, variableType in variableNameToTypeDict.items():
        if variableName not in dataframe.columns:
            raise KeyError("arithmetic.load_dataset(): The variable name '{}' was not found in variableNameToTypeDict: {}".format(variableName, variableNameToTypeDict))
        if variableType not in possibleTypes:
            raise TypeError("arithmetic.load_dataset(): The variable type '{}' was not found in possible types: {}".format(variableType, possibleTypes))
    if outputName not in dataframe.columns:
        raise KeyError("arithmetic.load_dataset(): The output name '{}' was not found in variableNameToTypeDict: {}".format(outputName, variableNameToTypeDict))
    if outputType not in possibleTypes:
        raise TypeError("arithmetic.load_dataset(): The output type '{}' was not found in possible types: {}".format(outputType, possibleTypes))

    dataset: List[Tuple[Dict[str, Union[float, bool] ], Union[float, bool]]] = []
    # Go through the lines
    for (index, row) in dataframe.iterrows():
        variableNameToValueDict: Dict[str, Union[float, bool] ] = {}
        for variableName, variableType in variableNameToTypeDict.items():
            value: str = row[variableName]
            if variableType == 'float':
                value = float(value)
            elif variableType == 'bool':
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                else:
                    raise ValueError("arithmetic.load_dataset(): Unsupported string for a bool: '{}'".format(value))
            else:
                raise ValueError("arithmetic.load_dataset(): Unsupported variable type '{}'".format(variableType))
            variableNameToValueDict[variableName] = value
        outputValue: str = row[outputName]
        if outputType == 'float':
            outputValue = float(outputValue)
        elif outputType == 'bool':
            if outputValue.lower() == 'true':
                outputValue = True
            elif outputValue.lower() == 'false':
                outputValue = False
            else:
                raise ValueError("arithmetic.load_dataset(): Unsupported string for a bool: '{}'".format(outputValue))
        else:
            raise ValueError("arithmetic.load_dataset(): Unsupported variable type '{}'".format(outputType))
        dataset.append((variableNameToValueDict, outputValue))
    return dataset
