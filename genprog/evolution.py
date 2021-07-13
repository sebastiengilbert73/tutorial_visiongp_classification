import logging
import genprog.core
from typing import Dict, List, Any, Set, Optional, Union, Tuple
import os
import abc
import random
import math
import copy
import xml.etree.ElementTree as ET
import sys
import statistics

class Population(abc.ABC):
    def __init__(self, individualsList: List[genprog.core.Individual]=None) -> None:
        self._individualsList: List[genprog.core.Individual] = []
        if individualsList is not None:
            self._individualsList = individualsList

    def LoadIndividuals(self, filepathPrefix: str) -> None:
        self._individualsList = []
        directory: str = os.path.dirname(filepathPrefix)
        filenamePrefix: str = os.path.basename(filepathPrefix)
        filenamesList: List[str] = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        for filename in filenamesList:
            if filename.startswith(filenamePrefix):
                individual: genprog.core.Individual = genprog.core.LoadIndividual(os.path.join(directory, filename))
                self._individualsList.append(individual)

    def SaveIndividuals(self, filepathPrefix: str) -> None:
        for individualNdx in range(len(self._individualsList)):
            filepath: str = filepathPrefix + '_' + str(individualNdx) + '.xml'
            #logging.debug ("Population.SaveIndividuals(): filepath = {}".format(filepath))
            self._individualsList[individualNdx].Save(filepath)

    def Generate(self,
                 numberOfIndividuals: int,
                 interpreter: genprog.core.Interpreter,
                 returnType: str,
                 levelToFunctionProbabilityDict: Dict[int, float],
                 proportionOfConstants: float,
                 constantCreationParametersList: List[Any],
                 variableNameToTypeDict: Dict[str, str],
                 functionNameToWeightDict: Dict[str, float]=None
                 ) -> None:
        self._individualsList = []
        for individualNdx in range(numberOfIndividuals):
            generatedIndividual = interpreter.CreateIndividual(
                returnType,
                levelToFunctionProbabilityDict,
                proportionOfConstants,
                constantCreationParametersList,
                variableNameToTypeDict,
                functionNameToWeightDict
            )
            self._individualsList.append(generatedIndividual)

    @abc.abstractmethod
    def EvaluateIndividualCosts(self, inputOutputTuplesList: List[ Tuple[ Dict[str, Any], Any ] ],
                                variableNameToTypeDict: Dict[str, str],
                                interpreter: genprog.core.Interpreter,
                                returnType: str,
                                weightForNumberOfElements: float) -> Dict[genprog.core.Individual, float]:
        pass

    def SelectTwoIndividualsWithCostInverse(self,
                                            inputOutputTuplesList: List[Tuple[Dict[str, Any], Any]],
                                            variableNameToTypeDict: Dict[str, str],
                                            interpreter: genprog.core.Interpreter,
                                            returnType: str,
                                            weightForNumberOfElements: float) -> Tuple[genprog.core.Individual, genprog.core.Individual]:
        maximumFitness: float = 1000.0
        individualToCostDict: Dict[genprog.core.Individual, float] = self.EvaluateIndividualCosts(
            inputOutputTuplesList,
            variableNameToTypeDict,
            interpreter,
            returnType,
            weightForNumberOfElements
        )
        rouletteList: List[float] = []
        fitnessSum = 0.0
        for individualNdx in range(len(self._individualsList)):
            cost: float = individualToCostDict[self._individualsList[individualNdx]]
            fitness: float = 0
            if cost < 1.0/maximumFitness:
                fitness = maximumFitness
            else:
                fitness = 1.0/cost
            rouletteList.append(fitness)
            fitnessSum += fitness
        # Normalize the roulette
        for individualNdx in range(len(rouletteList)):
            rouletteList[individualNdx] = rouletteList[individualNdx] / fitnessSum

        # Choose 1st individual
        selectedIndividual1: Optional[genprog.core.Individual] = None
        randomNbr: float = random.random()
        runningSum: float = 0.0
        theRandomNbrIsReached: bool = False
        for individualNdx in range(len(rouletteList)):
            runningSum += rouletteList[individualNdx]
            if runningSum >= randomNbr and not theRandomNbrIsReached:
                selectedIndividual1 = self._individualsList[individualNdx]
                theRandomNbrIsReached = True
        # Choose 2nd individual
        selectedIndividual2: Optional[genprog.core.Individual] = None
        randomNbr = random.random()
        runningSum = 0.0
        theRandomNbrIsReached = False
        for individualNdx in range(len(rouletteList)):
            runningSum += rouletteList[individualNdx]
            if runningSum >= randomNbr and not theRandomNbrIsReached:
                selectedIndividual2 = self._individualsList[individualNdx]
                theRandomNbrIsReached = True
        if selectedIndividual1 is None or selectedIndividual2 is None:
            raise ValueError("Population.SelectTwoIndividualsWithCostInverse(): One of the selected individuals is None...(?)")
        return (selectedIndividual1, selectedIndividual2)

    def CreateChild(self, parent1: genprog.core.Individual, parent2: genprog.core.Individual,
                    pivotElementReturnType: str, interpreter: genprog.core.Interpreter,
                    variableNameToTypeDict: Dict[str, str]) -> Optional[genprog.core.Individual]:
        child: genprog.core.Individual = copy.deepcopy(parent1)

        childElementsList: List[ET.Element] = genprog.core.ElementsWhoseReturnTypeIs(child, pivotElementReturnType,
                                                                                            interpreter,
                                                                                            variableNameToTypeDict)
        if len(childElementsList) == 0:
            return None
        childPivotElm: ET.Element = random.choice(childElementsList)
        parent2ElementsList: List[ET.Element] = genprog.core.ElementsWhoseReturnTypeIs(parent2, pivotElementReturnType,
                                                                                             interpreter,
                                                                                             variableNameToTypeDict)
        if len(parent2ElementsList) == 0:
            return None
        parent2ReplacementElm: ET.Element = random.choice(parent2ElementsList)

        parent_map: Dict[ET.Element, ET.Element] = {c: p for p in child._tree.iter() for c in p} # Cf. https://stackoverflow.com/questions/2170610/access-elementtree-node-parent-node
        childPivotParentElm = parent_map[childPivotElm]
        # What is the child index?
        childNdx: int = -1
        for candidateChildNdx in range(len(list(childPivotParentElm))):
            if childPivotParentElm[candidateChildNdx] == childPivotElm:
                childNdx = candidateChildNdx
        if childNdx == -1:
            raise ValueError("Population.CreateChild(): The child element could not be found in its parent children...(?)")
        # Replace the pivot
        childPivotParentElm[childNdx] = parent2ReplacementElm

        return child

    def Tournament(self, numberOfParticipants: int, individualToCostDict: Dict[genprog.core.Individual, float]) -> \
            genprog.core.Individual:
        if numberOfParticipants < 1:
            raise ValueError("Population.Tournament(): numberOfParticipants ({}) < 1".format(numberOfParticipants))
        participantsList: List[genprog.core.Individual] = []
        for participantNdx in range(numberOfParticipants):
            chosenParticipant = random.choice(self._individualsList)
            participantsList.append(chosenParticipant)
        lowestCost: float = sys.float_info.max
        champion: Optional[genprog.core.Individual] = None
        for participant in participantsList:
            if individualToCostDict[participant] < lowestCost:
                lowestCost = individualToCostDict[participant]
                champion = participant
        if champion is None:
            logging.warning("Population.Tournament(): champion is None. Returning participantsList[0]")
            return participantsList[0]
        return champion

    def NewGenerationWithTournament(self,
            inputOutputTuplesList: List[Tuple[Dict[str, Union[float, bool]], Union[float, bool]]],
            variableNameToTypeDict: Dict[str, str],
            interpreter: genprog.core.Interpreter,
            returnType: str,
            numberOfTournamentParticipants,
            mutationProbability,
            currentIndividualToCostDict: Optional[ Dict[genprog.core.Individual, float] ],
            proportionOfConstants: float,
            levelToFunctionProbabilityDict: Dict[int, float],
            functionNameToWeightDict: Optional[ Dict[ str, float]],
            constantCreationParametersList: List[Any],
            proportionOfNewIndividuals=0.05,
            weightForNumberOfElements=0.001,
            maximumNumberOfMissedCreationTrials: int =1000,
            ) -> Dict[genprog.core.Individual, float]:
        if currentIndividualToCostDict is None:
            currentIndividualToCostDict = self.EvaluateIndividualCosts(
                inputOutputTuplesList, variableNameToTypeDict, interpreter, returnType, weightForNumberOfElements
            )
        if mutationProbability < 0:
            mutationProbability = 0
        if mutationProbability > 1:
            mutationProbability = 1
        newGenerationList: List[genprog.core.Individual] = []

        typesList = interpreter.PossibleTypes()
        numberOfIndividuals: int = len(self._individualsList)
        # Generate new individuals
        numberOfNewIndividuals: int = round(proportionOfNewIndividuals * numberOfIndividuals)
        while len(newGenerationList) < numberOfNewIndividuals:
            newIndividual: genprog.core.Individual = interpreter.CreateIndividual(
                returnType,
                levelToFunctionProbabilityDict,
                proportionOfConstants,
                constantCreationParametersList,
                variableNameToTypeDict,
                functionNameToWeightDict
            )
            newGenerationList.append(newIndividual)

        numberOfMissedCreationTrials = 0
        while len(newGenerationList) < numberOfIndividuals:
            parent1: genprog.core.Individual = self.Tournament(numberOfTournamentParticipants, currentIndividualToCostDict)
            parent2: genprog.core.Individual = self.Tournament(numberOfTournamentParticipants, currentIndividualToCostDict)
            child = self.CreateChild(
                parent1,
                parent2,
                random.choice(typesList),
                interpreter,
                variableNameToTypeDict
            )
            if child is not None:
                # Determine if the child will mutate
                randomNbr = random.random()
                if randomNbr < mutationProbability:
                    child = interpreter.Mutate(
                        child,
                        levelToFunctionProbabilityDict,
                        proportionOfConstants,
                        functionNameToWeightDict,
                        constantCreationParametersList,
                        variableNameToTypeDict
                    )
                newGenerationList.append(child)
            else:
                numberOfMissedCreationTrials += 1
            if numberOfMissedCreationTrials >= maximumNumberOfMissedCreationTrials:
                raise RuntimeError("Population.NewGenerationWithTournament(): Reached the maximum number of missed creation trials: {}".format(maximumNumberOfMissedCreationTrials))
        self._individualsList = newGenerationList
        individualToCostDict = self.EvaluateIndividualCosts(
            inputOutputTuplesList, variableNameToTypeDict, interpreter, returnType, weightForNumberOfElements
        )
        return individualToCostDict

    def MedianCost(self, individualToCostDict: Dict[genprog.core.Individual, float] ) -> float:
        return statistics.median(individualToCostDict.values())

    def Champion(self, individualToCostDict: Dict[genprog.core.Individual, float]) -> Tuple[genprog.core.Individual, float]:
        lowestCost: float = sys.float_info.max
        champion: Optional[genprog.core.Individual] = None
        for individual, cost in individualToCostDict.items():
            if cost < lowestCost:
                lowestCost = cost
                champion = individual
        if champion is None:
            raise ValueError("Population.Champion(): champion is None... (?)")
        return (champion, lowestCost)

    def AverageEvaluation(self, inputsList: List[ Dict[str, Any] ],
                          interpreter: genprog.core.Interpreter,
                          variableNameToTypeDict: Dict[str, str],
                          expectedReturnType: str) -> List[ Tuple[ Dict[str, Any], Any] ]:
        inputToOutputList: List[ Tuple[ Dict[str, Any], Any] ] = []

        for inputDict in inputsList:
            outputSum = interpreter.Evaluate( self._individualsList[0],
                                              variableNameToTypeDict,
                                              inputDict,
                                              expectedReturnType)
            for indivudualNdx in range(1, len(self._individualsList)):
                outputSum = outputSum + interpreter.Evaluate( self._individualsList[indivudualNdx],
                                              variableNameToTypeDict,
                                              inputDict,
                                              expectedReturnType)
            averageOutput = 1.0/len(self._individualsList) * outputSum
            inputToOutputList.append((inputDict, averageOutput))
        return inputToOutputList

    def SumOfEvaluations(self, inputsList: List[ Dict[str, Any] ],
                          interpreter: genprog.core.Interpreter,
                          variableNameToTypeDict: Dict[str, str],
                          expectedReturnType: str) -> List[ Tuple[ Dict[str, Any], Any] ]:
        inputToOutputList: List[Tuple[Dict[str, Any], Any]] = []

        for inputDict in inputsList:
            outputSum = interpreter.Evaluate(self._individualsList[0],
                                             variableNameToTypeDict,
                                             inputDict,
                                             expectedReturnType)
            for indivudualNdx in range(1, len(self._individualsList)):
                outputSum = outputSum + interpreter.Evaluate(self._individualsList[indivudualNdx],
                                                             variableNameToTypeDict,
                                                             inputDict,
                                                             expectedReturnType)
            inputToOutputList.append((inputDict, outputSum))
        return inputToOutputList

"""
class ArithmeticsPopulation(Population): # An example to follow for other domains
    def EvaluateIndividualCosts(self,
                                inputOutputTuplesList: List[Tuple[Dict[str, Union[float, bool] ], Union[float, bool]]],
                                variableNameToTypeDict: Dict[str, str],
                                interpreter: genprog.core.Interpreter,
                                returnType: str,
                                weightForNumberOfElements: float=0.001) -> Dict[genprog.core.Individual, float]:
        penaltyForNoVariation = 1.0e6
        penaltyForOverflowError = 1.0e9
        individualToCostDict = {}
        if len(inputOutputTuplesList) == 0:
            raise ValueError("ArithmeticsPopulation.EvaluateIndividualCosts(): The list of input-output tuples is empty")
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
                    raise NotImplementedError("ArithmeticsPopulation.EvaluateIndividualCosts(): Not implemented return type '{}'".format(returnType))
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
                numberOfElements = NumberOfElements(individual._tree)
                individualToCostDict[individual] = individualToCostDict[individual] + weightForNumberOfElements * numberOfElements

        return individualToCostDict
"""

# Utilities
def NumberOfElements(tree: ET.ElementTree) -> int:
    count: int = 0
    for elm in tree.iter():
        count += 1
    return count

