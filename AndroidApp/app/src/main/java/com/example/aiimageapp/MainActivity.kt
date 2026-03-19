package com.example.aiimageapp

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Base64
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.unit.dp
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.ResponseBody
import org.json.JSONObject
import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            AppUI()
        }
    }
}

@Composable
fun AppUI() {
    var prompt by remember { mutableStateOf("") }
    var imageBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var loading by remember { mutableStateOf(false) }

    Column(modifier = Modifier.padding(16.dp)) {

        OutlinedTextField(
            value = prompt,
            onValueChange = { prompt = it },
            label = { Text("Enter prompt") },
            modifier = Modifier.fillMaxWidth()
        )

        Spacer(modifier = Modifier.height(10.dp))

        Button(
            onClick = {
                println("BUTTON CLICKED")
                loading = true
                generateImage(prompt) {
                    imageBitmap = it
                    loading = false
                }
            },
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("Generate Image")
        }

        Spacer(modifier = Modifier.height(20.dp))

        if (loading) {
            CircularProgressIndicator()
        }

        imageBitmap?.let {
            Image(
                bitmap = it.asImageBitmap(),
                contentDescription = null,
                modifier = Modifier.fillMaxWidth()
            )
        }
    }
}

fun generateImage(prompt: String, onResult: (Bitmap?) -> Unit) {

    val json = JSONObject()
    json.put("prompt", prompt)
    json.put("steps", 25)
    json.put("scale", 7.5)

    val body = json.toString()
        .toRequestBody("application/json".toMediaTypeOrNull())

    RetrofitClient.api.generateImage(body)
        .enqueue(object : Callback<ResponseBody> {

            override fun onResponse(
                call: Call<ResponseBody>,
                response: Response<ResponseBody>
            ) {
                try {
                    val res = response.body()?.string()
                    val obj = JSONObject(res!!)
                    val base64 = obj.getString("image")

                    val decodedBytes = Base64.decode(base64, Base64.DEFAULT)
                    val bitmap = BitmapFactory.decodeByteArray(decodedBytes, 0, decodedBytes.size)

                    onResult(bitmap)

                } catch (e: Exception) {
                    e.printStackTrace()
                    onResult(null)
                }
            }

            override fun onFailure(call: Call<ResponseBody>, t: Throwable) {
                t.printStackTrace()
                onResult(null)
            }
        })
}