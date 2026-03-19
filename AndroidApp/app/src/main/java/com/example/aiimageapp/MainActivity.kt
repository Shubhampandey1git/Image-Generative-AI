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

    fun saveImageToGallery(bitmap: android.graphics.Bitmap) {
        val filename = "AI_Image_${System.currentTimeMillis()}.png"

        val resolver = contentResolver
        val contentValues = android.content.ContentValues().apply {
            put(android.provider.MediaStore.MediaColumns.DISPLAY_NAME, filename)
            put(android.provider.MediaStore.MediaColumns.MIME_TYPE, "image/png")
            put(android.provider.MediaStore.MediaColumns.RELATIVE_PATH, "Pictures/AIImages")
        }

        val imageUri = resolver.insert(
            android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
            contentValues
        )

        imageUri?.let { uri ->
            val outputStream = resolver.openOutputStream(uri)
            outputStream?.use {
                bitmap.compress(android.graphics.Bitmap.CompressFormat.PNG, 100, it)
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            var imageBitmap by remember { mutableStateOf<android.graphics.Bitmap?>(null) }
            var loading by remember { mutableStateOf(false) }
            AppUI { bitmap ->
                saveImageToGallery(bitmap)
            }
        }
    }

}



@Composable
fun AppUI(onSave: (Bitmap) -> Unit) {
    var saved by remember { mutableStateOf(false) }
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
                contentDescription = "Generated Image",
                modifier = Modifier
                    .fillMaxWidth()
                    .height(300.dp)
            )
        }

        imageBitmap?.let {
            Spacer(modifier = Modifier.height(10.dp))

            Button(
                onClick = {
                    onSave(it)  // save when user clicks
                    saved = true
                          },
                modifier = Modifier.fillMaxWidth()
            ) {
                Text("Saved to Gallery")
            }

            Spacer(modifier = Modifier.height(20.dp))
            if (saved) {
                Text("Saved to Gallery")
            }

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