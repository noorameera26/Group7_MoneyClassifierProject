package main;

import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.ui.VertxUIServer;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.storage.FileStatsStorage;

import java.io.File;
import java.io.IOException;

public class LoadUIDL4J {
    public static void main(String[] args) throws IOException {

        //load the saved stats and visualize
        StatsStorage statsStorage = new FileStatsStorage(new File(System.getProperty("java.io.tmpdir"),"myNetworkTrainingStats.dl4j"));
        //VertxUIServer uiServer = VertxUIServer.getInstance(9001, false, null);
        UIServer uiServer = UIServer.getInstance();
        uiServer.attach(statsStorage);

        System.in.read();
    }
}
