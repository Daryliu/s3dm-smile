package com.smart3dmap.dto;

import com.smart3dmap.constants.CommonConstants;
import com.smart3dmap.constants.StatusConstants;

import java.io.Serializable;


public class BaseResponse implements Serializable {

	protected String status = CommonConstants.RESPONSE_STATUS_OK;

	protected String message = "";
	
	protected Integer code = 200;

	public String getStatus() {
		return status;
	}

	public void setStatus(String status) {
		this.status = status;
	}

	public String getMessage() {
		return message;
	}

	public void setMessage(String message) {
		this.message = message;
	}
	
	public Integer getCode() {
		return code;
	}

	public void setCode(Integer code) {
		this.code = code;
	}

	public void markFailure() {
		this.status = CommonConstants.RESPONSE_STATUS_FAILURE;
		this.code = StatusConstants.failure;
	}

	public static BaseResponse buildFailuaResponse(Exception e) {
		BaseResponse response = new BaseResponse();
		response.setStatus(CommonConstants.RESPONSE_STATUS_FAILURE);
		response.setMessage(e.getMessage());
		return response;
	}

	public static BaseResponse buildFailuaResponse(String error) {
		BaseResponse response = new BaseResponse();
		response.setStatus(CommonConstants.RESPONSE_STATUS_FAILURE);
		response.setMessage(error);
		return response;
	}
	
	public static BaseResponse buildFailuaResponse(String error,Integer code) {
		BaseResponse response = new BaseResponse();
		response.setStatus(CommonConstants.RESPONSE_STATUS_FAILURE);
		response.setCode(code);
		response.setMessage(error);
		return response;
	}
	
	public static BaseResponse buildSuccessResponse(){
		BaseResponse response = new BaseResponse();
		response.setStatus(CommonConstants.RESPONSE_STATUS_OK);
		return response;	
	}
	
	public static BaseResponse buildSuccessResponse(String message){
		BaseResponse response = new BaseResponse();
		response.setStatus(CommonConstants.RESPONSE_STATUS_OK);
		response.setMessage(message);
		return response;	
	}
    
	
}
