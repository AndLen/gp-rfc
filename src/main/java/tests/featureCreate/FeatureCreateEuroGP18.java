package tests.featureCreate;

import featureCreate.FeatureCreator;
import tests.Tests;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by lensenandr on 28/11/16.
 */
public class FeatureCreateEuroGP18 extends Tests {

    public List<String> getTestConfig() {
        Tests.main = FeatureCreator.class;
        FeatureCreator.MULTIVARIATE = false;
        return new ArrayList<>(Arrays.asList(
                "preprocessing=scale", "logPrefix=featureCreator/", "treeDepth=15", "featureMin=0", "featureMax=1", "numtrees=5", "featureCreateParamFile=src/main/java/gp/featureCreateMT.params"
                // "fitnessFunction=clusterFitness.LocumFitness",
        ));
    }


}
