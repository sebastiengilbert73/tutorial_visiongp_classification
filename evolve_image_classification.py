import logging
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--imagesDirectory', help="The images directory. Default: './data/squares_vs_circles/'", default='./data/squares_vs_circles/')
parser.add_argument('--classFilename', help="The filename of the classification file. Default: 'class.csv'", default='class.csv')
parser.add_argument('--numberOfIndividuals', help="The number of individuals. Default: 64", type=int, default=64)
parser.add_argument('--levelToFunctionProbabilityDict', help="The probability to generate a function, at each level. Default: '{0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1}'", default='{0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1}')
parser.add_argument('--proportionOfConstants', help='The probability to generate a constant, when a variable could be used. Default: 0', type=float, default=0)
parser.add_argument('--constantCreationParametersList', help="The parameters to use when creating constants: [minFloat, maxFloat, minInt, maxInt, width, height]. Default: '[-1, 1, 0, 255, 256, 256]'", default='[-1, 1, 0, 255, 256, 256]')
parser.add_argument('--primitivesFilepath', help="The filepath to the XML file for the primitive functions. Default: './vision_genprog/tasks/image_processing.xml'", default='./vision_genprog/tasks/image_processing.xml')
parser.add_argument('--outputDirectory', help="The output directory. Default: './outputs/'", default='./outputs/')
parser.add_argument('--numberOfGenerations', help="The number of generations to run. Default: 32", type=int, default=32)
parser.add_argument('--weightForNumberOfNodes', help="Penalty term proportional to the number of nodes. Default: 0.001", type=float, default=0.001)
parser.add_argument('--numberOfTournamentParticipants', help="The number of participants in selection tournaments. Default: 2", type=int, default=2)
parser.add_argument('--mutationProbability', help="The probability to mutate a child. Default: 0.1", type=float, default=0.1)
parser.add_argument('--proportionOfNewIndividuals', help="The proportion of randomly generates individuals per generation. Default: 0.1", type=float, default=0.1)
parser.add_argument('--maximumNumberOfMissedCreationTrials', help="The maximum number if missed creation trials. Default: 1000", type=int, default=1000)
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

levelToFunctionProbabilityDict = ast.literal_eval(args.levelToFunctionProbabilityDict)
constantCreationParametersList = ast.literal_eval(args.constantCreationParametersList)
image_shapeHW = (constantCreationParametersList[5], constantCreationParametersList[4])

def main():
    logging.info("create_classification_population.py main()")

    # Create the output directory
    if not os.path.exists(args.outputDirectory):
        os.makedirs(args.outputDirectory)


