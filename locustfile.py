from locust import HttpUser, task, between

class ImageNetUser(HttpUser):

    wait_time = between(0.1, 0.5)

    @task
    def predict(self):

        with open("test.jpg", "rb") as image:

            self.client.post(
                "/predict",
                files={
                    "file": (
                        "test.jpg",
                        image,
                        "image/jpeg"
                    )
                }
            )