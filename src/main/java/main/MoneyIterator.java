package main;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.*;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.common.primitives.Pair;
import org.slf4j.Logger;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
/*Functionalities of this class:
 * 1) */


public class MoneyIterator extends RecordReaderDataSetIterator {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(MoneyIterator.class);

    private static final String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
    private static final Random rng  = new Random(123);

    private static final int height = 224;
    private static final int width = 224;
    private static final int channels = 3;
    private static final int numClasses = 6;//6

    private static File mainDir;
    private static ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
    private static InputSplit trainData,testData;
    private static int batchSize;


    public MoneyIterator(RecordReader recordReader, int batchSize) {
        super(recordReader, batchSize);
    }

    public static RecordReaderDataSetIterator trainIterator() throws IOException {
        log.info("Creating train data set iterator...");
        return makeIterator(1, trainData, batchSize);
    }

    public static RecordReaderDataSetIterator testIterator() throws IOException {
        log.info("Creating test data set iterator...");
        return makeIterator(2, testData, 1);
    }

    /*public static RecordReaderDataSetIterator valIterator() throws IOException{
        log.info("Creating validation data set iterator...");
        return makeIterator(3, null, 1);
    }*/

    public static void setup(File parentDir, int batchSizeArg, int trainPerc){
        mainDir = parentDir;
        batchSize = batchSizeArg;
        FileSplit filesInDir = new FileSplit(mainDir, allowedExtensions, rng);

        BalancedPathFilter pathFilter = new BalancedPathFilter(rng, allowedExtensions, labelMaker);
        /*if (trainPerc >= 100) {
            throw new IllegalArgumentException("Percentage of data set aside for training has to be less than 100%.
            Test percentage = 100 - training percentage, has to be greater than 0");
        }*/

        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, trainPerc, 100-trainPerc);
        trainData = filesInDirSplit[0];
        testData = filesInDirSplit[1];
    }

    private static RecordReaderDataSetIterator makeIterator(int choice, InputSplit split, int batchSize) throws IOException {
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);

        if(choice == 1){
            log.info("Initializing image record reader with train data and image augmentation...");
            ImageTransform transform = transformImage();
            recordReader.initialize(split, transform);
        }
        else if(choice == 2){
            log.info("Initializing image record reader with test data...");
            recordReader.initialize(split);
        }
        /*else{
            log.info("Initializing image record reader with validation data...");
            recordReader.initialize(new FileSplit(mainDir, allowedExtensions, rng));
        }*/

        //for image record reader: label is always 1
        return new RecordReaderDataSetIterator(recordReader, batchSize, 1, numClasses);
    }

    private static ImageTransform transformImage(){
        ImageTransform crop = new CropImageTransform(rng, 15);
        ImageTransform rotate = new RotateImageTransform(rng,15);
        ImageTransform hFlip = new FlipImageTransform(1);

        List<Pair<ImageTransform, Double>> pipeline = Arrays.asList(
                new Pair<>(crop, 0.3),
                new Pair<>(rotate, 0.5),
                new Pair<>(hFlip, 0.5)
        );

        ImageTransform test = new PipelineImageTransform.Builder()
                .addImageTransform(new CropImageTransform(rng, 15), 15.)
                .addImageTransform(new RotateImageTransform(25)).build();

        return new PipelineImageTransform(pipeline);



    }
}
