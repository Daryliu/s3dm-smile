package com.smart3dmap.controller;

import com.smart3dmap.dto.BaseObjectResponse;
import com.smart3dmap.dto.BaseResponse;
import com.smart3dmap.dto.FitRequest;
import org.apache.commons.csv.CSVFormat;
import org.springframework.web.bind.annotation.*;
import smile.classification.KNN;
import smile.data.DataFrame;
import smile.io.Read;
import smile.math.TimeFunction;
import smile.stat.distribution.GaussianMixture;
import smile.validation.CrossValidation;
import smile.vq.Neighborhood;
import smile.vq.NeuralGas;
import smile.vq.SOM;

import java.io.IOException;
import java.net.URISyntaxException;
import java.text.ParseException;
import java.util.Arrays;

/**
 * Vector Quantization
 *
 * @author TianSu<ldy_rookie@163.com>
 * @Date 2024/9/12 9:59
 */
@RestController
@RequestMapping("/vectorQuantization")
public class VectorQuantizationController {

    //Self-Organizing Map，data/clustering/chameleon/t4.8k.txt
    @PostMapping("selfOrganizeMap")
    @ResponseBody
    public BaseResponse selfOrganizeMap(@RequestBody FitRequest request) throws IOException, URISyntaxException {
        var x = Read.csv(request.getArffFilePath(), CSVFormat.DEFAULT.withDelimiter(' ')).toArray();
        var epochs = 20;
        var lattice = SOM.lattice(20, 20, x);
        var som = new SOM(lattice,
                TimeFunction.constant(0.1),
                Neighborhood.Gaussian(1, x.length * epochs / 4));
        BaseObjectResponse baseObjectResponse = new BaseObjectResponse();
        baseObjectResponse.setData(som);
        return baseObjectResponse;
    }

    //Neural Gas，data/clustering/chameleon/t4.8k.txt
    @PostMapping("neuralGas")
    @ResponseBody
    public BaseResponse neuralGas(@RequestBody FitRequest request) throws IOException, URISyntaxException {
        var x = Read.csv(request.getArffFilePath(), CSVFormat.DEFAULT.withDelimiter(' ')).toArray();
        // 模型将训练 20 个完整的训练周期
        var epochs = 20;
        // 生成 400 个神经元
        var gas = new NeuralGas(NeuralGas.seed(400, x),
                TimeFunction.exp(0.3, x.length * epochs / 2),
                TimeFunction.exp(30, x.length * epochs / 8),
                TimeFunction.constant(x.length * 2));

        BaseObjectResponse baseObjectResponse = new BaseObjectResponse();
        baseObjectResponse.setData(gas);
        return baseObjectResponse;
    }

}
