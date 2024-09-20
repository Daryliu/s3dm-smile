package com.smart3dmap.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import smile.math.distance.Distance;

/**
 *
 *
 * @author TianSu<ldy_rookie@163.com>
 * @Date 2024/8/29 11:55
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class FitRequest {

    private String arffFilePath;
    private int k;
    private int datas;
    private Distance distance;

}
