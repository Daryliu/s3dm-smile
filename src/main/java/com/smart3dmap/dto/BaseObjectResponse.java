package com.smart3dmap.dto;

public class BaseObjectResponse extends BaseResponse{

	private Object tag;
	
	private Object data;

	public Object getData() {
		return data;
	}

	public void setData(Object data) {
		this.data = data;
	}

	public Object getTag() {
		return tag;
	}

	public void setTag(Object tag) {
		this.tag = tag;
	}	
}
