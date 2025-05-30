// static/js/script.js

document.addEventListener('DOMContentLoaded', function() {
    var submitBtn = document.getElementById('submit-btn');
    var imageInput = document.getElementById('image');
    var imageDisplay = document.querySelector('.Image-Display-Area img');


    if (submitBtn && imageInput && imageDisplay) {
        submitBtn.addEventListener('click', function(event) {
            event.preventDefault(); // Empêche le rechargement de la page

            var file = imageInput.files[0]; // Obtenir l'image sélectionnée

            if (!file) {
                alert('Please select an image first.');
                return;
            }

            // Lire l'image en tant qu'URL de données pour l'afficher immédiatement
            var reader = new FileReader();
            reader.onload = function(e) {
                imageDisplay.src = e.target.result; // Met à jour l'image affichée LOCAL
            };
            reader.readAsDataURL(file);

            // Optionnel : Envoyer l'image au serveur via fetch (mais ne pas modifier l'affichage)
            var formData = new FormData();
            formData.append('file', file);

            fetch('/Movie_Recommender/save_image_to_predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.message) {
                    alert(data.message);
                    // Ne pas modifier imageDisplay.src ici, car on affiche localement
                } else if (data.error) {
                    alert(data.error);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('An error occurred while uploading the image.');
            });
        });
    } else {
        console.error('One of the required elements is missing.');
    }
});



document.addEventListener('DOMContentLoaded', function() {
    var predictBtn = document.getElementById('predict-btn');
    var movieElements = [
        document.getElementById('movie1'),
        document.getElementById('movie2'),
        document.getElementById('movie3'),
        document.getElementById('movie4'),
        document.getElementById('movie5')
    ];

    if (predictBtn) {
        predictBtn.addEventListener('click', function() {
            fetch('/Movie_Recommender/ask_to_predict', { method: 'GET' })
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        alert("Error: " + data.error);
                        return;
                    }

                    // Vérifie si les images sont disponibles dans la réponse
                    if (data.Neighbors_Images && Array.isArray(data.Neighbors_Images)) {
                        // Parcours chaque bytecode base64 et l'affiche
                        data.Neighbors_Images.forEach((bytecode, index) => {
                            let movieElement = movieElements[index];
                            if (movieElement) {
                                // Décode le bytecode en format image
                                let imageUrl = 'data:image/png;base64,' + bytecode;
                                movieElement.src = imageUrl; // Met à jour la source de l'image
                            }
                        });
                    } else {
                        alert("No image data received.");
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('An error occurred while predicting.');
                });
        });
    } else {
        console.error("Predict button not found.");
    }
});
