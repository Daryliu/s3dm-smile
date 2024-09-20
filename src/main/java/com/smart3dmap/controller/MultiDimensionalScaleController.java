package com.smart3dmap.controller;

import com.smart3dmap.dto.BaseObjectResponse;
import com.smart3dmap.dto.BaseResponse;
import com.smart3dmap.dto.FitRequest;
import org.apache.commons.csv.CSVFormat;
import org.springframework.web.bind.annotation.*;
import smile.classification.KNN;
import smile.data.DataFrame;
import smile.io.Read;
import smile.manifold.MDS;
import smile.validation.CrossValidation;

import java.io.IOException;
import java.net.URISyntaxException;
import java.text.ParseException;

/**
 * Multi-Dimensional Scaling
 *
 * @author TianSu<ldy_rookie@163.com>
 * @Date 2024/9/12 11:37
 */
@RestController
@RequestMapping("/multiDimensionalScale")
public class MultiDimensionalScaleController {

    //Classical Multi-dimensional Scaling，data/mds/eurodist.txt
    @PostMapping("classical")
    @ResponseBody
    public BaseResponse classical(@RequestBody FitRequest request) throws IOException, URISyntaxException {
        var eurodist = Read.csv(request.getArffFilePath(), CSVFormat.DEFAULT.withDelimiter('\t'));
        var dist = eurodist.drop(0).toArray();
//        var citys = eurodist.stringVector(0).toArray();
        var x = MDS.of(dist, 2).coordinates;
        BaseObjectResponse baseObjectResponse = new BaseObjectResponse();
        baseObjectResponse.setData(x);
        return baseObjectResponse;
    }

}
