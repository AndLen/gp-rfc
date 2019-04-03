package tests.featureCreate;

import featureCreate.FeatureCreator;
import tests.Tests;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by lensenandr on 28/11/16.
 */
public class GPRFCSimMV extends Tests {

    public List<String> getTestConfig() {
        Tests.main = FeatureCreator.class;
        FeatureCreator.MULTIVARIATE = true;

        return new ArrayList<>(Arrays.asList(
                "preprocessing=scale", "logPrefix=featureCreatorSimMVGECCO18/", "treeDepth=10", "featureMin=0", "featureMax=1", "numtrees=10", "featureCreateParamFile=src/main/java/gp/featureCreateMTSimMV.params"
                // "fitnessFunction=clusterFitness.LocumFitness",
        ));
    }
}
