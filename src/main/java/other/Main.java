package other;

import data.Instance;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Collectors;

import static other.DatasetUtils.*;
import static other.Util.LOG;

/**
 * Created by lensenandr on 2/03/16.
 */
public class Main {

    public static final BetterProperties CONFIG = new BetterProperties();
    public static final int BOARD_WIDTH = 200;
    public static int RUN = 0;
    public static String[] METHOD_NAMES;


    public static class SetupSystem {
        private String[] args;
        private List<Instance> processedInstances;
        private int numFeatures;
        private int numClusters;

        public SetupSystem(String... args) {
            this.args = args;
        }

        public static List<Instance> getRawInstances(List<String> lines, String[] header) {
            String classLabelPosition = header[0];
            int numInitialFeatures = Integer.parseInt(header[1]);
            String splitString = ",";
            if (header[3].equals("space")) splitString = "\\s+";
            else if (header[3].equals("tab")) splitString = "\t";
            //Remove bad features
            List<Instance> rawInstances = getInstances(lines, classLabelPosition, numInitialFeatures, splitString);


            return rawInstances;
        }

        public List<Instance> getProcessedInstances() {
            return processedInstances;
        }

        public int getNumFeatures() {
            return numFeatures;
        }

        public int getNumClusters() {
            return numClusters;
        }

        public SetupSystem invoke() throws IOException {
            CONFIG.build(args);
            Util.initLogging();
            String dataset = CONFIG.getProperty("dataset");
            boolean doNNs = true;
            if (CONFIG.containsKey("doNNs")) {
                doNNs = CONFIG.getBoolean("doNNs");
            }
            return getSetupSystem(dataset, doNNs);
        }

        SetupSystem getSetupSystem(String dataset, boolean doNNs) throws IOException {
            List<String> lines;
            if (Util.IN_A_JAR) {
                Util.LOG.println("Inside a jar, loading config internally.");
                InputStream resource = Main.class.getClassLoader().getResourceAsStream(dataset);
                Util.LOG.println(resource);
                lines = new BufferedReader(new InputStreamReader(resource)).lines().collect(Collectors.toList());
            } else {
                lines = Files.readAllLines(Paths.get(System.getProperty("user.dir"), "/datasets", dataset));
            }


            String[] header = lines.get(0).split(",");

            numClusters = Integer.parseInt(header[2]);

            List<Instance> rawInstances = getRawInstances(lines, header);

            DatasetUtils.FEATURE_MIN = CONFIG.getInt("featureMin");
            DatasetUtils.FEATURE_MAX = CONFIG.getInt("featureMax");
            DatasetUtils.FEATURE_RANGE = (FEATURE_MAX - FEATURE_MIN);
            DatasetUtils.FEATURE_RANGE2 = (FEATURE_RANGE) * (FEATURE_RANGE);

            String preprocessingType = CONFIG.getProperty("preprocessing", "scale");
            switch (preprocessingType) {
                case "scale":
                    processedInstances = scaleInstances(rawInstances);
                    break;
                case "normalise":
                    processedInstances = normaliseInstances(rawInstances);
                    break;
                case "none":
                    processedInstances = rawInstances.stream().map(Instance::clone).collect(Collectors.toList());
                    break;
                default:
                    throw new IllegalArgumentException(preprocessingType);
            }
            DatasetUtils.initialise(processedInstances, doNNs);
            numFeatures = processedInstances.get(0).numFeatures();


            LOG.printf("System set up:%n%s%n%n", CONFIG.toString());
            return this;
        }
    }
}
