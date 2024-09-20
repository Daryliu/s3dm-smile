package com.smart3dmap.controller;

import com.smart3dmap.dto.BaseObjectResponse;
import com.smart3dmap.dto.BaseResponse;
import com.smart3dmap.dto.FitRequest;
import org.apache.commons.csv.CSVFormat;
import org.springframework.web.bind.annotation.*;
import smile.classification.KNN;
import smile.clustering.*;
import smile.clustering.linkage.CompleteLinkage;
import smile.data.DataFrame;
import smile.io.Read;
import smile.math.distance.EuclideanDistance;
import smile.util.SparseArray;
import smile.validation.CrossValidation;

import java.io.IOException;
import java.net.URISyntaxException;
import java.text.ParseException;

/**
 * 非监督学习--聚类
 *
 * @author TianSu<ldy_rookie@163.com>
 * @Date 2024/9/11 17:58
 */
@RestController
@RequestMapping("/cluster")
public class ClusterController {

    //--------------------Agglomerative Hierarchical Clustering，data/clustering/rem.txt-------------------------
    @PostMapping("hierarchicalCluster")
    @ResponseBody
    public BaseResponse hierarchicalCluster(@RequestBody FitRequest request) throws IOException, URISyntaxException {
        var x = Read.csv(request.getArffFilePath(), CSVFormat.DEFAULT.withDelimiter(' ')).toArray();
        var clusters = HierarchicalClustering.fit(CompleteLinkage.of(x));

        BaseObjectResponse baseObjectResponse = new BaseObjectResponse();
        baseObjectResponse.setData(clusters);
        return baseObjectResponse;
    }

    //-------------------K-Means,data/clustering/elongate.txt----------------------------------------
    @PostMapping("kMeans")
    @ResponseBody
    public BaseResponse kMeans(@RequestBody FitRequest request) throws IOException, URISyntaxException {
        var x = Read.csv(request.getArffFilePath(), CSVFormat.DEFAULT.withDelimiter('\t')).toArray();
        var clusters = PartitionClustering.run(20, () -> KMeans.fit(x, 2));

        BaseObjectResponse baseObjectResponse = new BaseObjectResponse();
        baseObjectResponse.setData(clusters);
        return baseObjectResponse;
    }

    //-------------------X-Means,data/clustering/rem.txt----------------------------------------
    @PostMapping("xMeans")
    @ResponseBody
    public BaseResponse xMeans(@RequestBody FitRequest request) throws IOException, URISyntaxException {
        var x = Read.csv(request.getArffFilePath(), CSVFormat.DEFAULT.withDelimiter(' ')).toArray();
        var clusters = XMeans.fit(x, 50);

        BaseObjectResponse baseObjectResponse = new BaseObjectResponse();
        baseObjectResponse.setData(clusters);
        return baseObjectResponse;
    }

    //-------------------G-Means,data/clustering/rem.txt----------------------------------------
    @PostMapping("gMeans")
    @ResponseBody
    public BaseResponse gMeans(@RequestBody FitRequest request) throws IOException, URISyntaxException {
        var x = Read.csv(request.getArffFilePath(), CSVFormat.DEFAULT.withDelimiter(' ')).toArray();
        var clusters = GMeans.fit(x, 50);

        BaseObjectResponse baseObjectResponse = new BaseObjectResponse();
        baseObjectResponse.setData(clusters);
        return baseObjectResponse;
    }


    //-------------------Sequential Information Bottleneck,data/libsvm/news20.dat---------------------------------------
    @PostMapping("sib")
    @ResponseBody
    public BaseResponse sib(@RequestBody FitRequest request) throws IOException, URISyntaxException {
        var data = Read.libsvm(request.getArffFilePath());
        var sparse = data.stream().map(i -> i.x()).toArray(SparseArray[]::new);
        var clusters = SIB.fit(sparse, 20, 100);

        BaseObjectResponse baseObjectResponse = new BaseObjectResponse();
        baseObjectResponse.setData(clusters);
        return baseObjectResponse;
    }

    //---------------------------------------CLARANS,data/clustering/elongate.txt---------------------------------------
    @PostMapping("clarans")
    @ResponseBody
    public BaseResponse clarans(@RequestBody FitRequest request) throws IOException, URISyntaxException {
        var x = Read.csv(request.getArffFilePath(), CSVFormat.DEFAULT.withDelimiter(' ')).toArray();
        var clusters = PartitionClustering.run(20, () -> CLARANS.fit(x, new EuclideanDistance(), 6, 10));

        BaseObjectResponse baseObjectResponse = new BaseObjectResponse();
        baseObjectResponse.setData(clusters);
        return baseObjectResponse;
    }

    //---------------------------------------dbscan,data/clustering/chameleon/t4.8k.txt---------------------------------
    @PostMapping("dbscan")
    @ResponseBody
    public BaseResponse dbscan(@RequestBody FitRequest request) throws IOException, URISyntaxException {
        var x = Read.csv(request.getArffFilePath(), CSVFormat.DEFAULT.withDelimiter(' ')).toArray();
        var clusters = DBSCAN.fit(x, 20, 10);

        BaseObjectResponse baseObjectResponse = new BaseObjectResponse();
        baseObjectResponse.setData(clusters);
        return baseObjectResponse;
    }

    //---------------------------------------denclue,data/clustering/rem.txt---------------------------------
    @PostMapping("denclue")
    @ResponseBody
    public BaseResponse denclue(@RequestBody FitRequest request) throws IOException, URISyntaxException {
        var x = Read.csv(request.getArffFilePath(), CSVFormat.DEFAULT.withDelimiter(' ')).toArray();
        var clusters = DENCLUE.fit(x, 1.0, 50);

        BaseObjectResponse baseObjectResponse = new BaseObjectResponse();
        baseObjectResponse.setData(clusters);
        return baseObjectResponse;
    }

    //---------------------------------------Spectral Clustering,data/clustering/sincos.txt-----------------------------
    @PostMapping("spectralCluster")
    @ResponseBody
    public BaseResponse spectralCluster(@RequestBody FitRequest request) throws IOException, URISyntaxException {
        var x = Read.csv(request.getArffFilePath(), CSVFormat.DEFAULT.withDelimiter('\t')).toArray();
        var clusters = SpectralClustering.fit(x, 2, 0.2);

        BaseObjectResponse baseObjectResponse = new BaseObjectResponse();
        baseObjectResponse.setData(clusters);
        return baseObjectResponse;
    }

    //--------------------------------Minimum Entropy Clustering,data/clustering/rem.txt-----------------------------
    @PostMapping("mec")
    @ResponseBody
    public BaseResponse mec(@RequestBody FitRequest request) throws IOException, URISyntaxException {
        var x = Read.csv(request.getArffFilePath(), CSVFormat.DEFAULT.withDelimiter(' ')).toArray();
        var clusters = MEC.fit(x, new EuclideanDistance(), 20, 2.0);

        BaseObjectResponse baseObjectResponse = new BaseObjectResponse();
        baseObjectResponse.setData(clusters);
        return baseObjectResponse;
    }

}
