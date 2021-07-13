import logging
import abc
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Set, Tuple, Optional, Union
import ast
import random
import xml.dom.minidom


class Individual():
    """
    Abstract class an individual class must inherit
    """

    def __init__(self, tree: ET.ElementTree):
        super().__init__()
        self._tree = tree

    def Save(self, filepath: str):
        rootElm: ET.Element = self._tree.getroot()
        treeStr: str = prettify(rootElm)
        treeStr = "".join([s for s in treeStr.splitlines(True) if s.strip()])
        with open(filepath, 'w') as file:
            file.write(treeStr)

    def Elements(self) -> List[ET.Element]:
        elementsList: List[ET.Element] = self.SubTreeElements(list(self._tree.getroot())[0])
        return elementsList

    def SubTreeElements(self, headElm: ET.Element) -> List[ET.Element]:
        subTreeElementsList: List[ET.Element] = [headElm]
        for childElm in list(headElm):
            childSubTreeElementsList: List[ET.Element] = self.SubTreeElements(childElm)
            subTreeElementsList.extend(childSubTreeElementsList)
        return subTreeElementsList




def prettify(elem): # Cf. https://stackoverflow.com/questions/17402323/use-xml-etree-elementtree-to-print-nicely-formatted-xml-files
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = xml.dom.minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="\t")

def LoadIndividual(filepath: str) -> Individual:
    tree = ET.parse(filepath)
    individual: Individual = Individual(tree)
    return individual

class FunctionSignature():
    def __init__(self, parameterTypesList: List[str], returnType: str) -> None:
        self._parameterTypesList = parameterTypesList
        self._returnType = returnType





class Interpreter(abc.ABC):
    def __init__(self, domainFunctionsTree: ET.ElementTree) -> None:
        super().__init__()
        _domainFunctionsTree = domainFunctionsTree

        # Create dictionaries from function name to parameters
        self._functionNameToSignatureDict: Dict[str, FunctionSignature] = {}

        # Check that each function name is unique
        root: ET.Element = _domainFunctionsTree.getroot()
        functionNamesSet: Set = set()
        for rootChild in root:
            #print("rootChild = {}".format(rootChild))
            #print ("rootChild.tag = {}".format(rootChild.tag))
            if rootChild.tag == 'function':
                functionNameElm: Optional[ET.Element] = rootChild.find('name')
                if functionNameElm is None:
                    raise ValueError("core.py Interpreter.__init__(): A function doesn't have a <name> element")
                functionName: Optional[str] = functionNameElm.text
                if functionName is None:
                    raise ValueError(
                        "core.py Interpreter.__init__(): A function have an empty <name> element")
                if functionName in functionNamesSet:
                    raise ValueError("core.py Interpreter.__init__(): The function name '{}' is encountered more than once in the domain functions tree".format(functionName))
                functionNamesSet.add(functionName)

                parameterTypesElm: Optional[ET.Element] = rootChild.find('parameter_types')
                if parameterTypesElm is None:
                    raise ValueError(
                        "core.py Interpreter.__init__(): The function {} doesn't have a <parameter_types> element".format(functionName))
                parameterTypesListStr: Optional[str] = parameterTypesElm.text
                if parameterTypesListStr is None:
                    raise ValueError("core.py Interpreter.__init__(): The function {} have an empty <parameter_types> element.".format(
                        functionName))
                parameterTypesListStr = parameterTypesListStr.replace(' ', '')
                parameterTypesListStr = parameterTypesListStr.replace('[', "['")
                parameterTypesListStr = parameterTypesListStr.replace(']', "']")
                parameterTypesListStr = parameterTypesListStr.replace(',', "','")
                #print ("core.py Interpreter.__init__(): parameterTypesListStr = {}".format(parameterTypesListStr))
                parameterTypesList = ast.literal_eval(parameterTypesListStr)
                #print ("core.py Interpreter.__init__(): parameterTypesList = {}".format(parameterTypesList))

                returnTypeElm: Optional[ET.Element] = rootChild.find('return_type')
                if returnTypeElm is None:
                    raise ValueError(
                        "core.py Interpreter.__init__(): The function {} doesn't have a <return_type> element".format(functionName))
                returnType: Optional[str] = returnTypeElm.text
                if returnType is None:
                    raise ValueError(
                        "core.py Interpreter.__init__(): The function {} have an empty <return_type> element".format(
                            functionName))

                signature: FunctionSignature = FunctionSignature(parameterTypesList, returnType)
                self._functionNameToSignatureDict[functionName] = signature
            else:
                raise ValueError("core.py Interpreter.__init__(): An child of the root element has tag '{}'".format(rootChild.tag))

    def TypeConverter(self, type: str, value: str) -> Any:
        if type == 'float':
            return float(value)
        elif type == 'int':
            return int(value)
        elif type == 'bool':
            if value.lower() in ['true', 'yes']:
                return True
            else:
                return False
        elif type == 'string':
            return value
        else:
            raise NotImplementedError("Interpreter.TypeConverter(): The type {} is not implemented".format(type))

    def Evaluate(self, individual: Individual, variableNameToTypeDict: Dict[str, str], variableNameToValueDict: Dict[str, Any],
                        expectedReturnType: Any) -> Any:
        individualRoot: ET.Element = individual._tree.getroot()
        if len(list(individualRoot)) != 1:
            raise ValueError("Interpreter.Evaluate(): The root has more than one children ({})".format(len(list(individualRoot))))
        headElement = list(individualRoot)[0]
        return self.EvaluateElement(headElement, variableNameToTypeDict, variableNameToValueDict, expectedReturnType)

    def EvaluateElement(self, element: ET.Element, variableNameToTypeDict: Dict[str, str], variableNameToValueDict: Dict[str, Any],
                        expectedReturnType: Any) -> Any:
        childrenList: List[ET.Element] = list(element)
        elementTag = element.tag

        if elementTag == 'constant':
            valueStr: Optional[str] = element.text
            if valueStr is None:
                raise ValueError("Interpreter.EvaluateElement(): A constant has no value")
            return self.TypeConverter(expectedReturnType, valueStr)
        elif elementTag == 'variable':
            variableName: Optional[str] = element.text
            if variableName is None:
                raise ValueError("Interpreter.EvaluateElement(): A variable has no name")
            if variableName not in variableNameToValueDict:
                raise KeyError("Interpreter.EvaluateElement(): Variable '{}' doesn't exist as a key in variableNameToValueDict".format(variableName))
            return variableNameToValueDict[variableName]
        else: # Function
            self.CheckIfSignatureMatches(elementTag, childrenList, variableNameToTypeDict, expectedReturnType)
            childrenEvaluationsList: List[Any] = []
            for childNdx in range(len(childrenList)):
                childExpectedReturnType = self._functionNameToSignatureDict[elementTag]._parameterTypesList[childNdx]
                childEvaluation: Any = self.EvaluateElement(childrenList[childNdx], variableNameToTypeDict, variableNameToValueDict, childExpectedReturnType)
                childrenEvaluationsList.append(childEvaluation)
            return self.FunctionDefinition(elementTag, childrenEvaluationsList)

    def EvaluateElements(self, element: ET.Element, variableNameToTypeDict: Dict[str, str], variableNameToValueDict: Dict[str, Any],
                        expectedReturnType: Any, elementToEvaluationDict: Dict[ET.Element, Any]=None) -> Dict[ET.Element, Any]:
        if elementToEvaluationDict is None:
            elementToEvaluationDict = {}
        childrenList: List[ET.Element] = list(element)
        elementTag = element.tag

        if elementTag == 'constant':
            valueStr: Optional[str] = element.text
            if valueStr is None:
                raise ValueError("Interpreter.EvaluateElement(): A constant has no value")
            elementToEvaluationDict[element] = self.TypeConverter(expectedReturnType, valueStr)
            return elementToEvaluationDict
        elif elementTag == 'variable':
            variableName: Optional[str] = element.text
            if variableName is None:
                raise ValueError("Interpreter.EvaluateElement(): A variable has no name")
            if variableName not in variableNameToValueDict:
                raise KeyError("Interpreter.EvaluateElement(): Variable '{}' doesn't exist as a key in variableNameToValueDict".format(variableName))
            elementToEvaluationDict[element] = variableNameToValueDict[variableName]
            return elementToEvaluationDict
        else: # Function
            self.CheckIfSignatureMatches(elementTag, childrenList, variableNameToTypeDict, expectedReturnType)
            childrenEvaluationsList: List[Any] = []
            for childNdx in range(len(childrenList)):
                childExpectedReturnType = self._functionNameToSignatureDict[elementTag]._parameterTypesList[childNdx]
                elementToEvaluationDict = self.EvaluateElements(childrenList[childNdx], variableNameToTypeDict, variableNameToValueDict, childExpectedReturnType,
                                                                elementToEvaluationDict)
                childrenEvaluationsList.append(elementToEvaluationDict[ childrenList[childNdx] ])
            elementToEvaluationDict[element] = self.FunctionDefinition(elementTag, childrenEvaluationsList)
            return elementToEvaluationDict

    def Backpropagate(self, headElement: ET.Element,
                      elementToEvaluationDict: Dict[ET.Element, Any],
                      elementToGradientDict: Dict[ET.Element, Any]=None) -> Dict[ET.Element, Any]:
        if elementToGradientDict is None: # The head of the tree: df1/df1 = 1.0
            elementToGradientDict = {headElement: 1.0}
        childrenList: List[ET.Element] = list(headElement)
        headElementTag = headElement.tag
        if headElement not in elementToGradientDict.keys():
            raise KeyError("Backpropagate(): The head element is not found in elementToGradientDict.keys()")
        headElementGradient = elementToGradientDict[headElement]
        argumentsList: List[Any] = [elementToEvaluationDict[child] for child in childrenList]
        if headElementTag != 'constant' and headElementTag != 'variable':
            childrenPartialDerivatives = self.FunctionDerivative(headElementTag, argumentsList)

            for childNdx in range(len(childrenList)):
                elementToGradientDict[childrenList[childNdx]] = headElementGradient * childrenPartialDerivatives[childNdx]
                elementToGradientDict = self.Backpropagate(childrenList[childNdx],
                                                           elementToEvaluationDict,
                                                           elementToGradientDict)
        return elementToGradientDict

    def VariableNameToGradientDict(self, headElement: ET.Element,
                                   elementToGradientDict: Dict[ET.Element, Any],
                                   variableNameToTypeDict: Dict[str, str]) -> Dict[str, Any]:
        variableNameToGradientDict: Dict[str, Any] = {}
        for element in elementToGradientDict.keys():
            if element.tag == 'variable':
                variableName: Optional[str] = element.text
                if variableName is None:
                    raise ValueError("Interpreter.VariableNameToGradientDict(): A variable has no name")
                if variableName in variableNameToGradientDict:
                    variableNameToGradientDict[variableName] = variableNameToGradientDict[variableName] + elementToGradientDict[element]
                else:
                    variableNameToGradientDict[variableName] = elementToGradientDict[element]
        return variableNameToGradientDict

    @abc.abstractmethod
    def FunctionDefinition(self, functionName: str, argumentsList: List[Any]) -> Any:
        pass

    def FunctionDerivative(self, functionName: str, argumentsList: List[Any]) -> List[Any]:
        pass

    @abc.abstractmethod
    def CreateConstant(self, returnType: str, parametersList: Optional[ List[Any] ] ) -> str:
        pass

    @abc.abstractmethod
    def PossibleTypes(self) -> List[str]:
        pass

    def CheckIfSignatureMatches(self, functionName: str, argumentsList: List[ET.Element], variableNameToTypeDict: Dict[str, str],
                                expectedReturnType) -> None:
        if functionName not in self._functionNameToSignatureDict:
            raise KeyError("Interpreter.CheckIfSignatureMatches(): The function name '{}' doesn't exist as a key in self._functionNameToSignatureDict".format(functionName))
        expectedSignature: FunctionSignature = self._functionNameToSignatureDict[functionName]
        if len(argumentsList) != len(expectedSignature._parameterTypesList):
            raise ValueError("Interpreter.CheckIfSignatureMatches(): For function '{}', len(argumentsList) ({}) != len(expectedSignature._parameterTypesList) ({})".format(
                functionName, len(argumentsList), len(expectedSignature._parameterTypesList)
            ))
        for argumentNdx in range(len(argumentsList)):
            argumentTag = argumentsList[argumentNdx].tag
            if argumentTag == 'variable':
                variableName: Optional[str] = argumentsList[argumentNdx].text
                if variableName is None:
                    raise ValueError("Interpreter.CheckIfSignatureMatches(): An variable argument has no name")
                if variableName not in variableNameToTypeDict:
                    raise KeyError("Interpreter.CheckIfSignatureMatches(): The variable name '{}' is not in variableNameToTypeDict".format(
                            variableName))
                argumentType: Optional[str] = variableNameToTypeDict[variableName]
            elif argumentTag == 'constant':
                argumentType = expectedSignature._parameterTypesList[argumentNdx] # Will be cast at evaluation
            else: # Function
                if argumentTag not in self._functionNameToSignatureDict:
                    raise KeyError("Interpreter.CheckIfSignatureMatches(): Function name '{}' doesn't exist as a key in self._functionNameToSignatureDict".format(argumentTag))
                argumentType = self._functionNameToSignatureDict[argumentTag]._returnType

            if argumentType != expectedSignature._parameterTypesList[argumentNdx]:
                raise ValueError("Interpreter.CheckIfSignatureMatches(): Expected paremter type {}, got argument type {}".format(expectedSignature._parameterTypesList[argumentNdx], argumentType))

        if expectedReturnType != expectedSignature._returnType:
            raise ValueError("Interpreter.CheckIfSignatureMatches(): The expected return type ({}) do not match the function signature return type ({})".format(expectedReturnType, expectedSignature._returnType))

    def FunctionsWhoseReturnTypeIs(self, returnType: str):
        functionsNamesList: List[str] = []
        for functionName, signature in self._functionNameToSignatureDict.items():
            if signature._returnType == returnType:
                functionsNamesList.append(functionName)
        return functionsNamesList

    def CreateElement(self, returnType: str,
                      level: int,
                      levelToFunctionProbabilityDict: Dict[int, float],
                      proportionOfConstants: float,
                      functionNameToWeightDict: Dict[str, float],
                      constantCreationParametersList: List[Any],
                      variableNameToTypeDict: Dict[str, str] ) -> ET.Element:
        # Is it a function?
        functionProbability: float = 0.0
        if level in levelToFunctionProbabilityDict:
            functionProbability = levelToFunctionProbabilityDict[level]
        if random.random() < functionProbability: # It is a function
            functionNamesList = self.FunctionsWhoseReturnTypeIs(returnType)
            # Normalize the probabilities
            functionNameToProbabilityDict: Dict[str, float] = {}
            weightsSum: float = 0.
            for functionName in functionNamesList:
                weightsSum += max(functionNameToWeightDict[functionName], 0.)
            if weightsSum == 0:
                raise ValueError("Interpreter.CreateElement(): returnType = {}; level = {}; The sum of weights is 0".format(returnType, level))
            for functionName in functionNamesList:
                functionNameToProbabilityDict[functionName] = max(functionNameToWeightDict[functionName], 0.0) / weightsSum
            randomNbr: float = random.random()
            runningSum: float = 0.0
            theRandomNbrIsReached: bool = False
            for functionName, probability in functionNameToProbabilityDict.items():
                runningSum += probability
                if runningSum >= randomNbr and not theRandomNbrIsReached:
                    element = ET.Element(functionName)
                    signature = self._functionNameToSignatureDict[functionName]
                    for parameterType in signature._parameterTypesList:
                        childElement = self.CreateElement(parameterType, level + 1, levelToFunctionProbabilityDict, proportionOfConstants,
                                                          functionNameToWeightDict, constantCreationParametersList,
                                                          variableNameToTypeDict)
                        element.append(childElement)
                    theRandomNbrIsReached = True
            return element
        else: # A variable or a constant
            # Try a variable
            itMustBeAConstant = False
            candidateVariableNamesList = []
            for variableName, type in variableNameToTypeDict.items():
                if type == returnType:
                    candidateVariableNamesList.append(variableName)
            if len(candidateVariableNamesList) == 0:
                itMustBeAConstant = True

            if itMustBeAConstant or random.random() < proportionOfConstants: # A constant
                element = ET.Element('constant')
                constantValueStr = self.CreateConstant(returnType, constantCreationParametersList)
                element.text = constantValueStr
                element.set('type', returnType)
                return element
            else: # A variable
                element = ET.Element('variable')
                chosenNdx: int = random.randint(0, len(candidateVariableNamesList) - 1)
                element.text = candidateVariableNamesList[chosenNdx]
                element.set('type', returnType)
                return element

    def CreateIndividual(self,
                         returnType: str,
                         levelToFunctionProbabilityDict: Dict[int, float],
                         proportionOfConstants: float,
                         constantCreationParametersList: List[Any],
                         variableNameToTypeDict: Dict[str, str],
                         functionNameToWeightDict: Dict[str, float]=None) -> Individual:
        if functionNameToWeightDict is None: # Create a uniform dictionary
            functionNameToWeightDict = {}
            for functionName, signature in self._functionNameToSignatureDict.items():
                functionNameToWeightDict[functionName] = 1.0
        root: ET.Element = ET.Element('individual')
        headElm: ET.Element = self.CreateElement(
            returnType,
            0,
            levelToFunctionProbabilityDict,
            proportionOfConstants,
            functionNameToWeightDict,
            constantCreationParametersList,
            variableNameToTypeDict
        )
        root.append(headElm)
        individual: Individual = Individual(ET.ElementTree(root))
        return individual

    def Mutate(self,
               individual: Individual,
               levelToFunctionProbabilityDict: Dict[int, float],
               proportionOfConstants: float,
               functionNameToWeightDict: Optional[Dict[str, float] ],
               constantCreationParametersList: List[Any],
               variableNameToTypeDict: Dict[str, str]
               ) -> Individual:
        if functionNameToWeightDict is None:
            functionNameToWeightDict = {}
            for functionName, signature in self._functionNameToSignatureDict.items():
                functionNameToWeightDict[functionName] = 1.0
        # Randomly select an element
        pivotElementReturnType: str = random.choice(self.PossibleTypes())
        candidateElementsList = ElementsWhoseReturnTypeIs(individual, pivotElementReturnType, self, variableNameToTypeDict)
        if len(candidateElementsList) == 0: # Don't mutate
            return individual
        pivotElement: ET.Element = random.choice(candidateElementsList)
        mutatedElement: ET.Element = self.CreateElement(pivotElementReturnType,
                                                        0,
                                                        levelToFunctionProbabilityDict,
                                                        proportionOfConstants,
                                                        functionNameToWeightDict,
                                                        constantCreationParametersList,
                                                        variableNameToTypeDict)

        parent_map: Dict[ET.Element, ET.Element] = {c: p for p in individual._tree.iter() for c in
                                                    p}  # Cf. https://stackoverflow.com/questions/2170610/access-elementtree-node-parent-node
        pivotParentElm = parent_map[pivotElement]
        # What is the child index?
        childNdx: int = -1
        for candidateChildNdx in range(len(list(pivotParentElm))):
            if pivotParentElm[candidateChildNdx] == pivotElement:
                childNdx = candidateChildNdx
        if childNdx == -1:
            raise ValueError(
                "Interpreter.Mutate(): The child element could not be found in its parent children...(?)")
        # Replace the pivot
        pivotParentElm[childNdx] = mutatedElement
        return individual

    def EpochOfTraining(self,
                        individual: Individual,
                        variableNameToTypeDict: Dict[str, str],
                        expectedReturnType: str,
                        trainingDataset: List[Tuple[Dict[str, Any], Any]],
                        learningRate: float) -> Individual:
        headElm: ET.Element = list(individual._tree.getroot())[0]
        for (trainingXDict, targetOutput) in trainingDataset:
            elementToEvaluationDict = self.EvaluateElements(
                headElm,
                variableNameToTypeDict,
                trainingXDict,
                expectedReturnType
            )
            delta = elementToEvaluationDict[headElm] - targetOutput
            """try:
                delta2: float = delta**2
            except:
                delta2 = delta"""

            elementToGradientDict = self.Backpropagate(
                headElm,
                elementToEvaluationDict
            )

            for element, gradient in elementToGradientDict.items():
                if element.tag == 'constant':
                    initialValue = elementToEvaluationDict[element]
                    try:
                        newValue = initialValue - delta * learningRate * gradient
                        element.text = str(newValue)
                    except: # The value type is not appropriate for updating. Ex.: bool
                        pass
        return individual


def ElementsWhoseReturnTypeIs(individual: Individual, returnType: str, interpreter: Interpreter, variableNameToTypeDict: Dict[str, str]):
    elementsList: List[ET.Element] = individual.Elements()
    elementsWithReturnTypeList: List[ET.Element] = []
    for element in elementsList:
        if element.tag == 'variable':
            variableName: Optional[str] = element.text
            if variableName is None:
                raise ValueError("ElementsWhoseReturnTypeIs(): A variable has no name")
            if variableName not in variableNameToTypeDict:
                raise KeyError("ElementsWhoseReturnTypeIs(): The variable name '{}' is not in variableNameToTypeDict")
            variableReturnType: str = variableNameToTypeDict[variableName]
            if variableReturnType == returnType:
                elementsWithReturnTypeList.append(element)
        elif element.tag == 'constant':
            constant_type = element.get('type')
            if constant_type == returnType:
                elementsWithReturnTypeList.append(element)
        else: # Function
            functionName: str = element.tag
            functionReturnType: str = interpreter._functionNameToSignatureDict[functionName]._returnType
            if functionReturnType == returnType:
                elementsWithReturnTypeList.append(element)
    return elementsWithReturnTypeList
