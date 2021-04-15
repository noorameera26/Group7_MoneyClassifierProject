package main;
/*
 * Copyright (c) 2019 Skymind Holdings Bhd.
 * Copyright (c) 2020 CertifAI Sdn. Bhd.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;
import org.datavec.api.records.metadata.RecordMetaDataURI;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.VertxUIServer;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.FileStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;
import org.nd4j.shade.guava.primitives.Ints;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_core.CV_8U;
import static org.bytedeco.opencv.global.opencv_core.CV_8UC3;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;

public class MoneyClassifier{
    private static final Logger log = LoggerFactory.getLogger(MoneyClassifier.class);
    private static ComputationGraph model;

    private static String modelFilename = "D:\\model\\run002.zip";
    //private static File modelDir = Paths.get(modelFilename).toFile();
    private static DataNormalization scaler;

    // parameters for vgg16 input image
    private static final int height = 224;
    private static final int width = 224;
    private static final int channels = 3;

    // parameters for the training phase
    private static int batchSize = 10;
    private static int nEpochs = 5; //40
    private static double learningRate = 1e-3;
    private static int nClasses = 6;
    private static List<String> labels;
    private static int seed = 123;

    public static void main(String[] args) throws Exception {

        /*1) Loading data from class path resource*/
        log.info("Loading data from class path resource...");
        File pathDir = new ClassPathResource("money-utm").getFile(); //place your dataset folder in the resources
        boolean exists = pathDir.exists();
        if(!exists){
            log.warn("File not found in the directory {}.", pathDir);
            log.info("Program will terminate...");
            System.exit(0);
        }

        /*2) Create iterators for training and testing*/
        MoneyIterator.setup(pathDir, batchSize,70);
        RecordReaderDataSetIterator trainIter = MoneyIterator.trainIterator();
        RecordReaderDataSetIterator testIter = MoneyIterator.testIterator();

        /*3) Normalize data with imagepreprocessing scaler*/
        scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(trainIter);
        trainIter.setPreProcessor(scaler);
        testIter.setPreProcessor(scaler);

        /*4) Get the labels for our classes*/
        labels = trainIter.getLabels();
        System.out.println(Arrays.toString(labels.toArray()));

        if (new File(modelFilename).exists()) {
            /*5) To perform inference: evaluate on testing data set using pre-load model.*/
            log.info("Loading model...");
            model = ModelSerializer.restoreComputationGraph(modelFilename);

            // User menu to perform selection
            switch (UserMenu.menu()) {
                case 2:
                    // perform one by one prediction and print evaluation metrics...
                    offlineValidationWithTestDataset(testIter);
                    break;
                case 3:
                    System.out.println("You chose not to do anything. Quitting the program...");
                    break;
                default:
                    // print evaluation metrics only...
                    evaluationTestDataSet(testIter);
            }
        }
        else {
            /*6) To perform training on the model.*/
            log.info("Building model...");

            // Load pretrained VGG16 model
            ComputationGraph pretrained = (ComputationGraph) VGG16.builder().build().initPretrained();
            log.info(pretrained.summary());

            // Transfer Learning steps - Model Configurations.
            FineTuneConfiguration fineTuneConf = getFineTuneConfiguration();

            // Transfer Learning steps - Modify prebuilt model's architecture for current scenario
            model = buildComputationGraph(pretrained, fineTuneConf);

            // Collect training stats from the network
            //VertxUIServer server = VertxUIServer.getInstance(9001, false, null);
            UIServer server = UIServer.getInstance();
            StatsStorage storage = new FileStatsStorage(new File(System.getProperty("java.io.tmpdir"),"myNetworkTrainingStats.dl4j"));
            server.attach(storage);
            model.setListeners(new ScoreIterationListener(1), new StatsListener(storage));

            // Perform training....
            log.info("Training model with...");
            log.info("...batch size: {},", batchSize);
            log.info("...number of epochs: {},", nEpochs);
            log.info("...learning rate: {}.", learningRate);
            for (int i = 1; i < nEpochs + 1; i++) {
                log.info("*** Starting epoch {} ***", i);
                long time = System.currentTimeMillis();
                trainIter.reset();
                while (trainIter.hasNext()) {
                    model.fit(trainIter.next());
                }
                time = System.currentTimeMillis() - time;
                log.info("*** Completed epoch {} within {}ms ***", i, time);
            }

            // Saving model...
            log.info("Saving model into {}", modelFilename);
            ModelSerializer.writeModel(model, modelFilename, true);
            //ModelSerializer.addNormalizerToModel(modelDir, scaler);

            // Perform evaluation...
            trainIter.reset();
            evaluationTrainDataSet(trainIter);
            evaluationTestDataSet(testIter);
        }

    }
    /*Method that perform the offline validation...*/
    private static void offlineValidationWithTestDataset(RecordReaderDataSetIterator test) throws InterruptedException, IOException {
        //NativeImageLoader imageLoader1 = new NativeImageLoader();
        NativeImageLoader imageLoader = new NativeImageLoader(height, width, channels); //load image
        CanvasFrame canvas = new CanvasFrame("Validate Test Dataset"); //canvas frame
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
        Mat convertedMat = new Mat();
        Mat convertedMat_big = new Mat();
        int count = 0;

        System.out.println("\n--------------------------------------------------------------------------------------------------------------------------------");
        System.out.println("\t\t\t\t\t\t\t\t\t\t\t\t\t VALIDATING TEST DATA SET");
        System.out.println("--------------------------------------------------------------------------------------------------------------------------------");
        while(test.hasNext()){
            test.next();
            count++;
        }
        System.out.println("Total data in the testing set: " + --count);
        test.reset();

        test.setCollectMetaData(true);
        count = 1;
        while (test.hasNext() && canvas.isVisible()) {
            System.out.println("\n- Validating test data set " + count++ + ".");
            DataSet ds = test.next();
            INDArray features = ds.getFeatures();

            Mat mat = imageLoader.asMat(features);
            mat.convertTo(convertedMat, CV_8UC3, 255, 0);
            int w = mat.cols() * 2;
            int h = mat.rows() * 2;
            resize(convertedMat, convertedMat_big, new Size(w, h));

            RecordMetaDataURI metadata = (RecordMetaDataURI) ds.getExampleMetaData().get(0);
            System.out.println("File URL: " + metadata.getURI());
            System.out.println("\na) Label: " + labels.get(Ints.asList(ds.getLabels().toIntVector()).indexOf(1)));
            getPredictions(features);

            canvas.showImage(converter.convert(convertedMat_big));
            canvas.waitKey();
        }
        test.reset();
        evaluationTestDataSet(test);

        canvas.dispose();
    }

    /*Methods that perform the evaluation...*/
    private static void evaluationTrainDataSet(RecordReaderDataSetIterator train){
        log.info("Validating train data set...");
        log.info("Train evaluation:");
        evaluationDataSet(train);
    }

    private static void evaluationTestDataSet(RecordReaderDataSetIterator test){
        log.info("Validating test data set...");
        log.info("Test evaluation:");
        evaluationDataSet(test);
    }

    private static void evaluationDataSet(RecordReaderDataSetIterator data){
        Evaluation evalTest = model.evaluate(data);
        log.info(evalTest.stats());
    }
    /*.........................*/

    /*Methods that perform prediction...*/
    private static void getPredictions(INDArray image) throws IOException {
        INDArray[] output = model.output(false, image);

        List<Prediction> predictions = decodePredictions(output[0], nClasses);
        System.out.println("b) Predictions: ");
        System.out.println(predictionsToString(predictions));
    }

    private static String predictionsToString(List<Prediction> predictions) {
        StringBuilder builder = new StringBuilder();
        for (Prediction prediction : predictions) {
            builder.append(prediction.toString());
            builder.append('\n');
        }
        return builder.toString();
    }

    private static List<Prediction> decodePredictions(INDArray encodedPredictions, int numPredicted) {
        List<Prediction> decodedPredictions = new ArrayList<>();
        int[] topX = new int[numPredicted];
        float[] topXProb = new float[numPredicted];

        int i = 0;
        for (INDArray currentBatch = encodedPredictions.getRow(0).dup(); i < numPredicted; ++i) {

            topX[i] = Nd4j.argMax(currentBatch, 1).getInt(0);
            topXProb[i] = currentBatch.getFloat(0, topX[i]);
            currentBatch.putScalar(0, topX[i], 0.0D);
            decodedPredictions.add(new Prediction(labels.get(topX[i]), (topXProb[i] * 100.0F)));
        }
        return decodedPredictions;
    }

    public static class Prediction {

        private String label;
        private double percentage;

        public Prediction(String label, double percentage) {
            this.label = label;
            this.percentage = percentage;
        }

        public void setLabel(final String label) {
            this.label = label;
        }

        public String toString() {

            return String.format("%s: %.2f ", this.label, this.percentage);
        }
    }
    /*................*/

    private static ComputationGraph buildComputationGraph(ComputationGraph pretrained, FineTuneConfiguration fineTuneConf) {
        double nonZeroBias = 0.1;
        double dropOut = 0.95;

        //Construct a new model with the intended architecture and print summary
        //For computation graph, you need to specify both nIn and nOut

        ComputationGraph vgg16Transfer = new TransferLearning.GraphBuilder(pretrained)
                .fineTuneConfiguration(fineTuneConf)
                .removeVertexKeepConnections("fc2")
                .addLayer("fc2",
                        new DenseLayer.Builder()
                                .nIn(4096)
                                .nOut(256)
                                .weightInit(new NormalDistribution(0,0.2*(2.0/(256+nClasses))))
                                //.activation(Activation.LEAKYRELU)
                                //.biasInit(nonZeroBias)
                                //.dropOut(dropOut)
                                .build(),
                        "fc1")
                .removeVertexKeepConnections("fc3")
                .addLayer("fc3",
                        new DenseLayer.Builder()
                                .nIn(256)
                                .nOut(64)
                                .weightInit(new NormalDistribution(0,0.2*(2.0/(256+nClasses))))
                                //.activation(Activation.LEAKYRELU)
                                //.biasInit(nonZeroBias)
                                //.dropOut(dropOut)
                                .build(),
                        "fc2")
                .removeVertexKeepConnections("predictions")
                .addLayer("predictions",
                        new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                .nIn(64)
                                .nOut(nClasses)
                                .weightInit(new NormalDistribution(0,0.2*(2.0/(224+nClasses)))) //This weight init dist gave better results than Xavier
                                .activation(Activation.SOFTMAX).build(),
                        "fc3")
                .setOutputs("predictions")
                .build();
        log.info(vgg16Transfer.summary());

        return vgg16Transfer;
    }

    private static FineTuneConfiguration getFineTuneConfiguration() {
        HashMap<Integer, Double> map = new HashMap<>();
        map.put(0, 8e-3);
        map.put(5, 8e-4);
        map.put(8, 8e-5);

        FineTuneConfiguration _FineTuneConfiguration = new FineTuneConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                //.gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)//divide gradient by L2 norm
                //.gradientNormalizationThreshold(1.0)
                .updater(new Adam(new MapSchedule(ScheduleType.EPOCH, map)))
                //.updater(new Nesterovs.Builder().learningRate(learningRate).momentum(Nesterovs.DEFAULT_NESTEROV_MOMENTUM).build())
                //.l2(0.00001) //dropout
                .activation(Activation.RELU)
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                .build();

        return _FineTuneConfiguration;
    }
}


