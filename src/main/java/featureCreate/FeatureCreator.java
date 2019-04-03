package featureCreate;

import data.Instance;
import ec.EvolutionState;
import ec.Evolve;
import ec.gp.GPIndividual;
import ec.gp.GPTree;
import ec.simple.SimpleStatistics;
import ec.util.Parameter;
import ec.util.ParameterDatabase;
import featureCreate.gp.FCProblemInterface;
import featureCreate.gp.FeatureCreatorStatistics;
import featureGrouping.MutualInformationMap;
import featureGrouping.ValuedFeature;
import other.Main;
import other.Util;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;

import static other.Main.CONFIG;
import static other.Util.LOG;

/**
 * Created by lensenandr on 29/08/17.
 */
public class FeatureCreator {

    public static final int SAMPLING_FRACTION = 1;
    public static final double EPSILON = 0.001; //0.1
    public static final double MIN_NOISE = EPSILON * 0.001;
    private static final boolean GP = true;
    private static final boolean ADD_NOISE = true;
    private static final boolean DO_ALL = true;
    public static String SOURCE_PREFIX;
    public static int[] FEAT_INDEX;// = 1;
    public static double[] xVals;// = {.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0};
    public static double[] noisyXVals;
    //[featureIndex][featureLength]
    public static double[][] multipleXVals;
    public static double[][] multipleNoisyXVals;
    public static double[][][] noisyXValsPerSubPop;
    public static double baseMI;
    public static double baseMultiInfo;
    public static boolean MULTIVARIATE = true;
    public static int NUM_SOURCE_MV = -1;
    public static int FEAT_PER_SUBPOP;
    public static int[][] SUBPOP_INDICIES;
    private static Integer[][] sortedSourceIndicies;

    public static void main(String[] args) throws IOException {
        //  if (args.length == 1) {
        //  String configPath = Paths.get(System.getProperty("user.dir"), "/src/tests/tests.config").toString();
        //   Main.RUN = args.length == 2 ? Integer.parseInt(args[1]) : 0;
        String[] args2 = Arrays.copyOf(args, args.length + 1);
        args2[args.length] = "doNNs=true";
        Main.SetupSystem system = new Main.SetupSystem(args2).invoke();
        List<Instance> processedInstances = system.getProcessedInstances();

        int numFeatures = system.getNumFeatures();
        List<ValuedFeature> valuedFeatures = instancesToValuedFeatures(processedInstances, numFeatures);

        if (NUM_SOURCE_MV == -1) {
            //Want at least 4 different r.f sets?
            //At least 2 sources, no more than 5...for now.
            NUM_SOURCE_MV = Math.max(2, Math.min(valuedFeatures.size() / 4, 5));
            LOG.printf("Source features set to %d.\n", NUM_SOURCE_MV);
        }
        if (DO_ALL) {

            List<List<String>> allLines = new ArrayList<>();
            int numSources = MULTIVARIATE ? NUM_SOURCE_MV : 1;
            for (int i = 0; (i + numSources) <= numFeatures; i += numSources) {
                FEAT_INDEX = new int[numSources];
                for (int j = 0; j < numSources; j++) {
                    FEAT_INDEX[j] = i + j;
                }
                initSourcePrefix();
                List<String> redundantFeatures = createRedundantFeatures(processedInstances, valuedFeatures);
                allLines.add(redundantFeatures);

            }
            StringBuilder sb = new StringBuilder();
            for (int j = 0; j < allLines.size(); j++) {
                sb.append(allLines.get(j).get(0)).append(", ");
            }
            sb.append("class\n");

            int numRFs = allLines.get(0).size();
            for (int i = 1; i < numRFs; i++) {
                for (int j = 0; j < allLines.size(); j++) {
                    sb.append(allLines.get(j).get(i)).append(", ");
                }
                sb.append(processedInstances.get(i - 1).getClassLabel());
                sb.append("\n");
            }
            LOG.println(sb.toString());
            String dataset = CONFIG.getProperty("dataset").replaceAll("/", "");
            Path outDir = Util.LOG.DIRECTORY.resolve(String.format("%s-%s/", dataset, new SimpleDateFormat("yyyy-MM-dd-HH-mm-ss").format(new Date())));
            //String outDir = String.format("/home/lensenandr/phd/phd/fg/%s/%s/", new SimpleDateFormat("yyyy-MM-dd-HH-mm-ss").format(new Date()), dataset);
            Files.createDirectories(outDir);
            String outPrefix = String.format("gp%s%dRF-%.1f", dataset, numRFs, FCProblemInterface.MIN_SOURCE_MI);
            String outCSV = outPrefix + ".csv";
            Path csvPath = outDir.resolve(outCSV);
            Files.write(csvPath, Collections.singletonList(sb.toString()));

            runScript(outDir, csvPath, "plotRF.R");
            runScript(outDir, csvPath, "plotGradientRF.R");

        } else {
            FEAT_INDEX = new int[]{0};
            initSourcePrefix();
            createRedundantFeatures(processedInstances, valuedFeatures);

        }

    }

    private static void initSourcePrefix() {
        StringBuilder sourcePrefix = new StringBuilder();
        for (int fI : FEAT_INDEX) {
            sourcePrefix.append(fI).append("-");
        }
        //Remove final '-'
        sourcePrefix.deleteCharAt(sourcePrefix.length() - 1);
        SOURCE_PREFIX = sourcePrefix.toString();
    }

    static void runScript(Path outDir, Path csvPath, String rScript) throws IOException {
        String format = String.format("/usr/pkg/bin/Rscript /home/lensenandr/phd/conferences/eurogp2018/%s %s %s", rScript, csvPath.toString(), outDir.toString());

        //System.out.println(format);
        try {
            Process exec = Runtime.getRuntime().exec(format);
            System.out.println(new BufferedReader(new InputStreamReader(exec.getInputStream()))
                    .lines().collect(Collectors.joining("\n")));
            System.err.println(new BufferedReader(new InputStreamReader(exec.getErrorStream()))
                    .lines().collect(Collectors.joining("\n")));
            exec.waitFor();//
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        // .getInputStream();
    }

    static List<String> createRedundantFeatures(List<Instance> processedInstances, List<ValuedFeature> valuedFeatures) {
        Random random = new Random(0);
        multipleXVals = new double[FEAT_INDEX.length][];
        multipleNoisyXVals = new double[FEAT_INDEX.length][];

        for (int fI = 0; fI < FEAT_INDEX.length; fI++) {
            int index = FEAT_INDEX[fI];
            ValuedFeature feature = valuedFeatures.get(index);

            //Don't sort -- that way preserve order and mapping of instances to noisy features.
            double[] values = Arrays.copyOf(feature.values, feature.values.length);
            //Arrays.sort(values);
            multipleXVals[fI] = values;
            LOG.println(feature.featureID + " : " + Arrays.toString(values));
            if (SAMPLING_FRACTION < 1) {
                multipleXVals[fI] = new double[values.length / SAMPLING_FRACTION];
                for (int j = 0; j < values.length; j += SAMPLING_FRACTION) {
                    multipleXVals[fI][j / SAMPLING_FRACTION] = values[j];
                }
            }
            // }
            if (ADD_NOISE) {
                multipleNoisyXVals[fI] = new double[multipleXVals[fI].length];
                for (int j = 0; j < multipleXVals[fI].length; j++) {
                    //Better results with duplicate values (e.g. RW), removes 0s (div by 0 less likely )
                    double noise = MIN_NOISE + (random.nextDouble() * (EPSILON - MIN_NOISE));
                    multipleNoisyXVals[fI][j] = multipleXVals[fI][j] + noise;
                }

                LOG.println(Arrays.toString(multipleXVals[fI]));
                LOG.println(Arrays.toString(multipleNoisyXVals[fI]));
            } else {
                multipleNoisyXVals[fI] = new double[multipleXVals[fI].length];
                for (int j = 0; j < multipleXVals[fI].length; j++) {
                    multipleNoisyXVals[fI][j] = multipleXVals[fI][j] + EPSILON;
                }
            }

        }

        if (FEAT_INDEX.length == 1) {
            LOG.println("Single source feature being used: " + FEAT_INDEX[0]);
            xVals = multipleXVals[0];
            noisyXVals = multipleNoisyXVals[0];
            baseMI = MutualInformationMap.getMutualInformation(xVals, xVals);
            LOG.println("Base MI: " + baseMI);
        } else {
            LOG.println(FEAT_INDEX.length + " source features being used: " + Arrays.toString(FEAT_INDEX));

        }


        StringBuilder sb = new StringBuilder();


        if (GP) {
            EvolutionState state = doGP();
            GPIndividual individual = (GPIndividual) ((SimpleStatistics) state.statistics).best_of_run[0];
            GPTree[] trees = individual.trees;
            FCProblemInterface problem = (FCProblemInterface) state.evaluator.p_problem;
            double[][] outputs = problem.getAllOutputs(state, 0, individual);


            List<String> fileOutput = new ArrayList<>();
            for (int fI : FEAT_INDEX) {
                sb.append(String.format("F%s, ", fI));

            }
            sb.delete(sb.length() - 2, sb.length());

            for (int j = 0; j < trees.length; j++) {
                sb.append(String.format(", F%s%c", SOURCE_PREFIX, FeatureCreatorStatistics.getCharToUse(j)));
            }
            //  sb.append(",class");
            fileOutput.add(sb.toString());
            for (int i = 0; i < processedInstances.size(); i++) {
                sb = new StringBuilder();
                Instance instance = processedInstances.get(i);
                for (int fI : FEAT_INDEX) {
                    sb.append(instance.getFeatureValue(fI)).append(", ");

                }
                for (int j = 0; j < trees.length; j++) {
                    sb.append(String.format("%f", outputs[j][i])).append(", ");

                }
                //sb.append(instance.getClassLabel());
                sb.delete(sb.length() - 2, sb.length());
                fileOutput.add(sb.toString());
            }
            fileOutput.forEach(LOG::println);
            //FeatureCreatorStatistics.OutputEvaluator outputEvaluator = new FeatureCreatorStatistics.OutputEvaluator(outputs);
            //outputEvaluator.invoke();
            // Files.write(Paths.get(String.format("gp%sF%s%.2f-%.2f-scaleInstance.csv", CONFIG.getProperty("dataset").replaceAll("/", ""), SOURCE_PREFIX, outputEvaluator.getWorstSourceMI(), outputEvaluator.getWorstSharedMI())), fileOutput);

            return fileOutput;
        }
        return null;
    }

    static EvolutionState doGP() {
        String paramName = Main.CONFIG.getProperty("featureCreateParamFile");
        ParameterDatabase parameters = Evolve.loadParameterDatabase(new String[]{"-file", paramName});

        int numtrees = Main.CONFIG.getInt("numtrees");
        int numSpecies = SUBPOP_INDICIES == null ? 1 : SUBPOP_INDICIES.length;

        if (numtrees != -1) {
            Util.LOG.printf("%d trees\n", numtrees);
            for (int j = 0; j < numSpecies; j++) {
                parameters.set(new Parameter("pop.subpop." + j + ".species.ind.numtrees"), Integer.toString(numtrees));
                for (int i = 0; i < numtrees; i++) {
                    parameters.set(new Parameter("pop.subpop." + j + ".species.ind.tree." + i), "ec.gp.GPTree");
                    parameters.set(new Parameter("pop.subpop." + j + ".species.ind.tree." + i + ".tc"), "tc0");
                }
            }
            if (MULTIVARIATE) {
                baseMI = MutualInformationMap.getMultiVarMutualInformation(multipleXVals, multipleXVals, false);
            } else {
                double[][] fakeOutputs = new double[numtrees + 1][];
                for (int i = 0; i < numtrees + 1; i++) {
                    fakeOutputs[i] = xVals;
                }
                baseMultiInfo = MutualInformationMap.getMultiInformation(fakeOutputs, false);
            }
        }
        LOG.println("Base MI: " + baseMI);
        LOG.println("Base MultiInfo: " + baseMultiInfo);

        if (Main.CONFIG.containsKey("treeDepth")) {
            int treeDepth = Main.CONFIG.getInt("treeDepth");
            Util.LOG.printf("Tree depth: %d\n", treeDepth);

            parameters.set(new Parameter("gp.koza.xover.maxdepth"), "" + treeDepth);
            //         parameters.set(new Parameter("gp.koza.xover.maxdepth"),""+treeDepth);

            //this is important as otherwise xover can make it too big...
            parameters.set(new Parameter("gp.koza.xover.maxsize"), "" + treeDepth);

            parameters.set(new Parameter("gp.koza.grow.max-depth"), "" + treeDepth);
            parameters.set(new Parameter("gp.koza.full.max-depth"), "" + treeDepth);
            parameters.set(new Parameter("gp.koza.half.max-depth"), "" + treeDepth);

        }


        int processors = Runtime.getRuntime().availableProcessors();
        int threads = Math.max(processors / 2, 1);
        Util.LOG.printf("%d processors, using %d threads\n", processors, threads);

        parameters.set(new Parameter("evalthreads"), "" + threads);
        parameters.set(new Parameter("stat.file"), Util.LOG.ECJ_OUT + Main.RUN + "F" + SOURCE_PREFIX);
        parameters.set(new Parameter("stat.front"), Util.LOG.PARETO_OUT);
        int seed = ThreadLocalRandom.current().nextInt();
        for (int i = 0; i < threads; i++) {
            parameters.set(new Parameter("seed." + i), Integer.toString(seed));
            seed++;
        }


        EvolutionState state = Evolve.initialize(parameters, 0);
        state.run(EvolutionState.C_STARTED_FRESH);
        return state;

    }

    public static Integer[][] getSortedSourceIndicies() {
        //Lazy cache
        if (sortedSourceIndicies == null) {
            sortedSourceIndicies = new Integer[multipleXVals.length][];
            for (int i = 0; i < multipleXVals.length; i++) {
                double[] sourceVals = Arrays.copyOf(multipleXVals[i], multipleXVals[i].length);

                sortedSourceIndicies[i] = new Integer[sourceVals.length];
                for (int j = 0; j < sourceVals.length; j++) {
                    sortedSourceIndicies[i][j] = j;
                }
                Arrays.sort(sortedSourceIndicies[i], (o1, o2) -> Double.compare(sourceVals[o1], sourceVals[o2]));

            }

        }
        return sortedSourceIndicies;
    }

    public static List<ValuedFeature> instancesToValuedFeatures(List<Instance> processedInstances, int numFeatures) {
        double[][] featureValues = new double[numFeatures][processedInstances.size()];
        for (int i = 0; i < processedInstances.size(); i++) {
            Instance instance = processedInstances.get(i);
            final double[] instanceVals = instance.featureValues;
            for (int j = 0; j < instanceVals.length; j++) {
                featureValues[j][i] = instanceVals[j];
            }
        }
        List<ValuedFeature> valuedFeatures = new ArrayList<>();
        for (int i = 0; i < featureValues.length; i++) {
            valuedFeatures.add(new ValuedFeature(featureValues[i], i));
        }


        return valuedFeatures;
    }
}
