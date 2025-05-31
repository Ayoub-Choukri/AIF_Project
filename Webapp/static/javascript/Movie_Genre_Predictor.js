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

            // Effacer la prédiction précédente
            document.getElementById('predicted-genre').textContent = "";
            // Effacer les probabilités précédentes
            const genres = [
                "action", "animation", "comedy", "documentary", "drama",
                "fantasy", "horror", "romance", "science-fiction", "thriller"
            ];
            
            genres.forEach(genre => {
                const cell = document.getElementById(`${genre}-probability`);
                if (cell) {
                    cell.textContent = "0%";
                }
            });

            // Lire l'image en tant qu'URL de données pour l'afficher immédiatement
            var reader = new FileReader();
            reader.onload = function(e) {
                imageDisplay.src = e.target.result; // Met à jour l'image affichée LOCAL
            };
            reader.readAsDataURL(file);

            // Optionnel : Envoyer l'image au serveur via fetch
            var formData = new FormData();
            formData.append('file', file);

            // Construire dynamiquement l’URL avec le service
            fetch('/Movie_Genre_Predictor/save_image_to_predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.message) {
                    alert(data.message);
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
    if (predictBtn) {
        predictBtn.addEventListener('click', function() {
            fetch('/Movie_Genre_Predictor/ask_to_predict', { method: 'GET' })
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        alert("Error: " + data.error);
                        return;
                    }

                    // Mettre à jour le genre prédit
                    document.getElementById('predicted-genre').textContent = data.Best_Prediction;

                    // Mettre à jour les probabilités par genre
                    Object.entries(data.Probas).forEach(([genre, probability]) => {
                        let genreId = genre.toLowerCase().replace(/\s/g, '-') + '-probability'; // Convertit en ID HTML
                        let probabilityCell = document.getElementById(genreId);
                        if (probabilityCell) {
                            probabilityCell.textContent = probability + '%';
                        }
                    });
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



