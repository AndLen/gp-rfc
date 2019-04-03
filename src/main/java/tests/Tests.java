package tests;

/**
 * Created by lensenandr on 4/04/16.
 */

import org.junit.FixMethodOrder;
import org.junit.Test;
import org.junit.experimental.categories.Category;
import org.junit.runners.MethodSorters;
import other.Main;

import java.lang.management.ManagementFactory;
import java.lang.management.ThreadMXBean;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

@FixMethodOrder(MethodSorters.NAME_ASCENDING)
public class Tests {
    public static Class<?> main = Main.class;

    private void runTestWithConfig(String configFile) {
        String configPath = Paths.get(System.getProperty("user.dir"), "/src/main/java/tests/config", configFile).toString();
        runTestWithArgs(configPath);

    }

    private void runTestWithArgs(String... supplied) {
        ArrayList<String> args = new ArrayList<>();
        Collections.addAll(args, supplied);
        args.addAll(getTestConfig());
        args.addAll(Arrays.asList(SingleJUnitTestRunner.PARAMS));
        System.out.println(args);
        try {
            Method main = Tests.main.getMethod("main", String[].class);
            main.invoke(null, (Object) args.toArray(new String[args.size()]));
        } catch (NoSuchMethodException e) {
            throw new Error(e);
        } catch (InvocationTargetException | IllegalAccessException e) {
            e.printStackTrace();
        }
        ThreadMXBean threadMXBean = ManagementFactory.getThreadMXBean();
        for (long id : threadMXBean.getAllThreadIds()) {
            long threadCpuTime = threadMXBean.getThreadCpuTime(id);
            System.out.println(threadCpuTime);
        }
    }

    private void runTestWithDataset(String datasetFile) {
        String configPath = Paths.get(System.getProperty("user.dir"), "/src/main/java/tests/config/tests.config").toString();
        runTestWithArgs(configPath, datasetFile);

    }

    public List<String> getTestConfig() {
        return new ArrayList<>(Arrays.asList("featureSubsetForFitness=false", "preprocessing=scale"));
    }

    @Test
    @Category(FeatureLearnerTests.class)
    public void a_irisTest() {
        runTestWithConfig("iris.config");
    }

    @Test
    @Category(FeatureLearnerTests.class)
    public void b_wineTest() {
        runTestWithConfig("wine.config");
    }

    @Category({ManyFeatureTests.class, FeatureLearnerTests.class})
    @Test
    public void c_moveLibrasTest() {
        runTestWithConfig("movement_libras.config");
    }

    @Category(FeatureLearnerTests.class)
    @Test
    public void d_dermatologyTest() {
        runTestWithConfig("dermatology.config");
    }

    @Category(FeatureLearnerTests.class)
    @Test
    public void e_breastCancerTest() {
        runTestWithConfig("breast-cancer-wisconsin.config");
    }

    @Category(FeatureLearnerTests.class)
    @Test
    public void f_imageSegmentationTest() {
        runTestWithConfig("image-segmentation.config");
    }

    @Test
    public void nhs_spiralTest() {
        runTestWithConfig("spiral.config");
    }

    @Test
    public void nhs_aggregationTest() {
        runTestWithConfig("aggregation.config");
    }

    @Test
    public void nhs_compoundTest() {
        runTestWithConfig("compound.config");
    }

    @Test
    public void nhs_d31Test() {
        runTestWithConfig("d31.config");
    }

    @Test
    public void nhs_flameTest() {
        runTestWithConfig("flame.config");
    }

    @Test
    public void nhs_jainTest() {
        runTestWithConfig("jain.config");
    }

    @Test
    public void nhs_pathbasedTest() {
        runTestWithConfig("pathbased.config");
    }

    @Test
    public void nhs_r15Test() {
        runTestWithConfig("r15.config");
    }

    @Test
    public void z_10d10ctest() {
        runTestWithConfig("10d10c.config");
    }

    @Test
    public void z_10d20ctest() {
        runTestWithConfig("10d20c.config");
    }

    @Test
    public void z_10d40ctest() {
        runTestWithConfig("10d40c.config");
    }

    @Test
    public void z_50d10ctest() {
        runTestWithConfig("ellipsoid.50d10c.config");
    }

    @Test
    public void z_50d20ctest() {
        runTestWithConfig("ellipsoid.50d20c.config");
    }

    @Test
    public void z_50d40ctest() {
        runTestWithConfig("ellipsoid.50d40c.config");
    }

    @Test
    public void z_100d10ctest() {
        runTestWithConfig("ellipsoid.100d10c.config");
    }

    @Test
    public void z_100d20ctest() {
        runTestWithConfig("ellipsoid.100d20c.config");
    }

    @Test
    public void z_100d40ctest() {
        runTestWithConfig("ellipsoid.100d40c.config");
    }

    @Test
    public void z2_10d10ctest() {
        runTestWithConfig("10d10cE.config");
    }

    @Test
    public void z2_10d20ctest() {
        runTestWithConfig("10d20cE.config");
    }

    @Test
    public void z2_10d40ctest() {
        runTestWithConfig("10d40cE.config");
    }

    @Test
    public void z2_10d100ctest() {
        runTestWithConfig("10d100cE.config");
    }

    @Test
    public void z2_10d1000ctest() {
        runTestWithConfig("10d1000cE.config");
    }

    @Test
    public void z2_1000d10ctest() {
        runTestWithConfig("1000d10c.config");
    }

    @Test
    public void z2_1000d20ctest() {
        runTestWithConfig("1000d20c.config");
    }

    @Test
    public void z2_1000d40ctest() {
        runTestWithConfig("1000d40c.config");
    }

    @Test
    public void z2_1000d100ctest() {
        runTestWithConfig("1000d100c.config");
    }

    @Test
    public void z3_1000d10cGaussiantest() {
        runTestWithConfig("1000d10cGaussian.config");
    }

    @Test
    public void z3_1000d100cGaussiantest() {
        runTestWithConfig("1000d100cGaussian.config");
    }

    @Test
    public void sparse_50d10ctest() {
        runTestWithConfig("ellipsoid.50d10cSparse.config");
    }

    @Test
    public void sparse_100d10ctest() {
        runTestWithConfig("ellipsoid.100d10cSparse.config");
    }

    @Test
    public void sparse_1000d10ctest() {
        runTestWithConfig("1000d10cSparse.config");
    }

    @Test
    public void sparse_1000d20ctest() {
        runTestWithConfig("1000d20cSparse.config");
    }

    @Test
    public void sparse_1000d40ctest() {
        runTestWithConfig("1000d40cSparse.config");
    }

    @Test
    public void sparse_1000d100ctest() {
        runTestWithConfig("1000d100cSparse.config");
    }

    @Test
    public void subspace_one() {
        runTestWithDataset("dataset=subspace/oneNoise.subspace");
    }

    @Test
    public void subspace_big() {
        runTestWithDataset("dataset=subspace/big.subspace");
    }


    @Test
    public void featureGroupOne_test() {
        runTestWithDataset("dataset=featureGroup/one.fg");
    }


    @Test
    @Category(FeatureLearnerTests.class)
    public void bioinformaticTest() {
        runTestWithDataset("dataset=bioinformaticYeung/4rep_low_noise/syn_sine_2_mult1.andrew.csv");
    }

//    @Test
//    public void featureGroupTOX_test()  {
//        runTestWithDataset("dataset=featureGroup/TOX_171.fg");
//    }

    @Test
    public void in10d10cTest() {
        runTestWithDataset("dataset=featureGroup/10d10c.0.fg");
    }

    //
    @Test
    @Category(FeatureLearnerTests.class)
    public void mfatTest() {
        runTestWithConfig("mfat.config");
    }
//
//    @Test
//    public void letterRecognitionTest() {
//        runTestWithConfig("letter-recognition.config");
//    }
//

    @Test
    @Category(FeatureLearnerTests.class)
    public void vehicleTest() {
        runTestWithDataset("dataset=vehicle.data");
    }

    @Test
    @Category(FeatureLearnerTests.class)
    public void germanTest() {
        runTestWithDataset("dataset=classification/german.data");
    }

    @Test
    @Category(FeatureLearnerTests.class)
    public void ionosphereTest() {
        runTestWithDataset("dataset=classification/ionosphere.data");
    }

    @Test
    @Category(FeatureLearnerTests.class)
    public void lungCancerTest() {
        runTestWithDataset("dataset=classification/lung-cancer.data");
    }

    @Category({ManyFeatureTests.class})
    @Test
    public void muskTest() {
        runTestWithDataset("dataset=classification/musk-clean1.data");
    }

    @Test
    @Category(FeatureLearnerTests.class)
    public void sonarTest() {
        runTestWithDataset("dataset=classification/sonar.data");
    }

    @Category({ManyFeatureTests.class, FeatureLearnerTests.class})
    @Test
    public void colonTest() {
        runTestWithDataset("dataset=manyFeatures/colon.data");
    }

    @Category({ManyFeatureTests.class, FeatureLearnerTests.class})
    @Test
    public void isoletTest() {
        runTestWithDataset("dataset=manyFeatures/Isolet.data");
    }

    @Category({ManyFeatureTests.class, FeatureLearnerTests.class})
    @Test
    public void lungDiscreteTest() {
        runTestWithDataset("dataset=manyFeatures/lung_discrete.data");
    }

    @Category({ManyFeatureTests.class, FeatureLearnerTests.class})
    @Test
    public void madelonTest() {
        runTestWithDataset("dataset=manyFeatures/madelon.data");
    }

    @Category({ManyFeatureTests.class, FeatureLearnerTests.class})
    @Test
    public void coil20Test() {
        runTestWithDataset("dataset=manyFeatures/COIL20.data");
    }

    @Category({ManyFeatureTests.class, FeatureLearnerTests.class})
    @Test
    public void yaleTest() {
        runTestWithDataset("dataset=manyFeatures/Yale.data");
    }

    @Category(FeatureLearnerTests.class)
    @Test
    public void uspsTest() {
        runTestWithDataset("dataset=manyFeatures/USPS.data");
    }

    @Category(FeatureLearnerTests.class)
    @Test
    public void epilepticTest() {
        runTestWithDataset("dataset=manyFeatures/epileptic.data");
    }

    @Category(FeatureLearnerTests.class)
    @Test
    public void mnist_1k_23Test() {
        runTestWithDataset("dataset=manyFeatures/mnist_train_1k_23.data");
    }

    @Category(FeatureLearnerTests.class)
    @Test
    public void mnist_1k_all() {
        runTestWithDataset("dataset=manyFeatures/mnist_train_1k.data");
    }

    @Category(FeatureLearnerTests.class)
    @Test
    public void olivettiTest() {
        runTestWithDataset("dataset=manyFeatures/olivetti.data");
    }


    public interface ManyFeatureTests {
    }

    public interface FeatureLearnerTests {

    }


}
