package com.example.aiimageapp

import okhttp3.RequestBody
import okhttp3.ResponseBody
import retrofit2.Call
import retrofit2.http.Body
import retrofit2.http.POST

interface ApiService {
    @POST("generate")
    fun generateImage(@Body body: RequestBody): Call<ResponseBody>
}