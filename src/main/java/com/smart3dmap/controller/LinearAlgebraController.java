package com.smart3dmap.controller;

import com.smart3dmap.dto.BaseObjectResponse;
import com.smart3dmap.dto.BaseResponse;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.bind.annotation.RestController;
import smile.interpolation.*;

/**
 * Linear Algebra
 *
 * @author TianSu<ldy_rookie@163.com>
 * @Date 2024/9/12 11:57
 */
@RestController
@RequestMapping("/linearAlgebra")
public class LinearAlgebraController {

    //LinearInterpolation
    @PostMapping("linearInterpolation")
    @ResponseBody
    public BaseResponse linearInterpolation(double[] x, double[] y) {
        double[][] points = new double[x.length][2];
        for (int i = 0; i < x.length; i++) {
            points[i][0] = x[i];
            points[i][1] = y[i];
        }
        var linear = new LinearInterpolation(x, y);
        double[][] data = new double[61][2];
        for (int i = 0; i < data.length; i++) {
            data[i][0] = i * 0.1;
            data[i][1] = linear.interpolate(data[i][0]);
        }

        BaseObjectResponse baseObjectResponse = new BaseObjectResponse();
        baseObjectResponse.setData(data);
        return baseObjectResponse;
    }

    //CubicSplineInterpolation
    @PostMapping("cubicSplineInterpolation")
    @ResponseBody
    public BaseResponse cubicSplineInterpolation(double[] x, double[] y) {
        double[][] points = new double[x.length][2];
        for (int i = 0; i < x.length; i++) {
            points[i][0] = x[i];
            points[i][1] = y[i];
        }
        var cubic = new CubicSplineInterpolation1D(x, y);
        double[][] data = new double[61][2];
        for (int i = 0; i < data.length; i++) {
            data[i][0] = i * 0.1;
            data[i][1] = cubic.interpolate(data[i][0]);
        }

        BaseObjectResponse baseObjectResponse = new BaseObjectResponse();
        baseObjectResponse.setData(data);
        return baseObjectResponse;
    }

    //KrigingInterpolation1D
    @PostMapping("krigingInterpolation")
    @ResponseBody
    public BaseResponse krigingInterpolation(double[] x, double[] y) {
        double[][] points = new double[x.length][2];
        for (int i = 0; i < x.length; i++) {
            points[i][0] = x[i];
            points[i][1] = y[i];
        }
        var kriging = new KrigingInterpolation1D(x, y);
        double[][] data = new double[61][2];
        for (int i = 0; i < data.length; i++) {
            data[i][0] = i * 0.1;
            data[i][1] = kriging.interpolate(data[i][0]);
        }

        BaseObjectResponse baseObjectResponse = new BaseObjectResponse();
        baseObjectResponse.setData(data);
        return baseObjectResponse;
    }

    //RBFInterpolation1D
    @PostMapping("rbfInterpolation")
    @ResponseBody
    public BaseResponse rbfInterpolation(double[] x, double[] y) {
        double[][] points = new double[x.length][2];
        for (int i = 0; i < x.length; i++) {
            points[i][0] = x[i];
            points[i][1] = y[i];
        }
        var rbf = new RBFInterpolation1D(x, y, new smile.math.rbf.GaussianRadialBasis());
        double[][] data = new double[61][2];
        for (int i = 0; i < data.length; i++) {
            data[i][0] = i * 0.1;
            data[i][1] = rbf.interpolate(data[i][0]);
        }

        BaseObjectResponse baseObjectResponse = new BaseObjectResponse();
        baseObjectResponse.setData(data);
        return baseObjectResponse;
    }

    //Shepard Interpolation
    @PostMapping("shepardInterpolation")
    @ResponseBody
    public BaseResponse shepardInterpolation(double[] x, double[] y) {
        double[][] points = new double[x.length][2];
        for (int i = 0; i < x.length; i++) {
            points[i][0] = x[i];
            points[i][1] = y[i];
        }
        var shepard = new ShepardInterpolation1D(x, y, 3);
        double[][] data = new double[61][2];
        for (int i = 0; i < data.length; i++) {
            data[i][0] = i * 0.1;
            data[i][1] = shepard.interpolate(data[i][0]);
        }

        BaseObjectResponse baseObjectResponse = new BaseObjectResponse();
        baseObjectResponse.setData(data);
        return baseObjectResponse;
    }

}
