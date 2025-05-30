// static/js/script.js

document.addEventListener("DOMContentLoaded", function () {
    document.getElementById("redirect-btn").addEventListener("click", function(event) {
        event.preventDefault();

        // Récupérer la valeur de l'option sélectionnée
        var selectedService = document.getElementById("services").value;

        if (selectedService) {
            // Envoyer une requête POST pour définir le service
            fetch('/define_service', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ Service: selectedService })
            })
            .then(response => response.json())
            .then(data => {
                console.log("Service défini avec succès:", data);
                // Rediriger vers la page de prédiction
                window.location.href = "/prediction_page";
            })
            .catch(error => {
                console.error("Erreur lors de la définition du service:", error);
                alert("Une erreur est survenue. Veuillez réessayer.");
            });
        } else {
            alert("Veuillez sélectionner un service d'abord.");
        }
    });
});

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

            fetch('/save_image_to_predict', {
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

    if (predictBtn) {
        predictBtn.addEventListener('click', function() {
            fetch('/ask_to_predict', { method: 'GET' })
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



