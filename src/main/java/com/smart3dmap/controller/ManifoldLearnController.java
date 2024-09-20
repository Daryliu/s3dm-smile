package com.smart3dmap.controller;

import com.smart3dmap.dto.BaseObjectResponse;
import com.smart3dmap.dto.BaseResponse;
import com.smart3dmap.dto.FitRequest;
import org.apache.commons.csv.CSVFormat;
import org.springframework.web.bind.annotation.*;
import smile.feature.extraction.PCA;
import smile.io.Read;
import smile.manifold.*;

import java.io.IOException;
import java.net.URISyntaxException;
import java.util.Arrays;

/**
 * Manifold Learning
 *
 * @author TianSu<ldy_rookie@163.com>
 * @Date 2024/9/12 11:40
 */
@RestController
@RequestMapping("/manifoldLearn")
public class ManifoldLearnController {

    //Iso map,data/manifold/swissroll.txt
    @PostMapping("isoMap")
    @ResponseBody
    public BaseResponse isoMap(@RequestBody FitRequest request) throws IOException, URISyntaxException {
        var swissroll = Read.csv(request.getArffFilePath(), CSVFormat.DEFAULT.withDelimiter('\t')).toArray();
        var map = IsoMap.of(Arrays.copyOf(swissroll, 2000), 7);

        BaseObjectResponse baseObjectResponse = new BaseObjectResponse();
        baseObjectResponse.setData(map);
        return baseObjectResponse;
    }

    //LLE,data/manifold/swissroll.txt
    @PostMapping("lle")
    @ResponseBody
    public BaseResponse lle(@RequestBody FitRequest request) throws IOException, URISyntaxException {
        var swissroll = Read.csv(request.getArffFilePath(), CSVFormat.DEFAULT.withDelimiter('\t')).toArray();
        var map = LLE.of(Arrays.copyOf(swissroll, 2000), 8);

        BaseObjectResponse baseObjectResponse = new BaseObjectResponse();
        baseObjectResponse.setData(map);
        return baseObjectResponse;
    }

    //Laplacian Eigenmap,data/manifold/swissroll.txt
    @PostMapping("laplacianEigenmap")
    @ResponseBody
    public BaseResponse laplacianEigenmap(@RequestBody FitRequest request) throws IOException, URISyntaxException {
        var swissroll = Read.csv(request.getArffFilePath(), CSVFormat.DEFAULT.withDelimiter('\t')).toArray();
        var map = LaplacianEigenmap.of(Arrays.copyOf(swissroll, 1000), 10, 2, 25.0);

        BaseObjectResponse baseObjectResponse = new BaseObjectResponse();
        baseObjectResponse.setData(map);
        return baseObjectResponse;
    }

    //TSNE,data/mnist/mnist2500_X.txt
    @PostMapping("tSne")
    @ResponseBody
    public BaseResponse tSne(@RequestBody FitRequest request) throws IOException, URISyntaxException {
        var format = CSVFormat.DEFAULT.withDelimiter(' ');
        var mnist = Read.csv("data/mnist/mnist2500_X.txt", format).toArray();
        var pca = PCA.fit(mnist).getProjection(50);
        var X = pca.apply(mnist);
        var perplexity = 20;
        var tsne = new TSNE(X, 2, perplexity, 200, 1000);

        BaseObjectResponse baseObjectResponse = new BaseObjectResponse();
        baseObjectResponse.setData(tsne);
        return baseObjectResponse;
    }

    //UMAP,data/mnist/mnist2500_X.txt
    @PostMapping("uMap")
    @ResponseBody
    public BaseResponse uMap(@RequestBody FitRequest request) throws IOException, URISyntaxException {
        var format = CSVFormat.DEFAULT.withDelimiter(' ');
        var mnist = Read.csv(request.getArffFilePath(), format).toArray();
        var map = UMAP.of(mnist);

        BaseObjectResponse baseObjectResponse = new BaseObjectResponse();
        baseObjectResponse.setData(map);
        return baseObjectResponse;
    }

}
