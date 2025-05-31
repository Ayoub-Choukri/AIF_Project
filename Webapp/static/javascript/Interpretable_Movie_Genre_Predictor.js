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

            // Effacer la prédiction précédente
            document.getElementById('predicted-genre').textContent = "";
            // Mettre à jour les probabilités par genre
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

            // Optionnel : Envoyer l'image au serveur via fetch
            var formData = new FormData();
            formData.append('file', file);

            

            // Construire dynamiquement l’URL avec le service
            fetch('/Interpretable_Movie_Genre_Predictor/save_image_to_predict', {
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
            fetch('/Interpretable_Movie_Genre_Predictor/ask_to_predict', { method: 'GET' })
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

// 🔁 Fonction pour définir la technique d'interprétation
async function setInterpretationTechnique(technique) {
    const response = await fetch(`/Interpretable_Movie_Genre_Predictor/set_interpretation_technique`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ technique: technique })
    });

    const result = await response.json();
    if (!response.ok) {
        alert("Erreur lors du choix de la technique : " + result.error);
    } else {
        console.log("Technique d'interprétation définie :", technique);
    }
}
async function interpretImage() {
    console.log("📡 Requête GET vers /ask_to_interpret...");

    try {
        const response = await fetch("/Interpretable_Movie_Genre_Predictor/ask_to_interpret", {
            method: "GET"
        });

        if (!response.ok) {
            const result = await response.json();
            alert("Erreur d’interprétation : " + result.error);
            return;
        }

        const result = await response.json();

        const imgData = result.interpreted_image;

        if (!imgData || !imgData.startsWith("data:image")) {
            alert("Format d'image reçu invalide.");
            return;
        }

        // Image originale affichée en haut
        const originalImg = document.querySelector(".Image-Display-Area img");

        // Image d'interprétation
        const interpretImg = document.getElementById("interpret-image");
        interpretImg.src = imgData;
        interpretImg.style.display = "block";

        if (originalImg) {
            // Récupérer la taille visible actuelle (en pixels)
            const width = originalImg.clientWidth;
            const height = originalImg.clientHeight;

            // Appliquer la taille EXACTE à l’image d’interprétation
            interpretImg.style.width = width + "px";
            interpretImg.style.height = height + "px";

            console.log(`✅ Image d’interprétation redimensionnée en ${width}x${height}px.`);
        } else {
            console.warn("⚠️ Impossible de récupérer l'image originale pour la taille.");
            interpretImg.style.width = "300px";  // fallback
            interpretImg.style.height = "auto";
        }

    } catch (error) {
        console.error("Erreur JS :", error);
        alert("Erreur d’interprétation (réseau ou parsing) : " + error.message);
    }
}




// 📎 Lier les boutons
document.addEventListener('DOMContentLoaded', () => {
    document.getElementById("confirm-method-btn").addEventListener("click", async () => {
        const technique = document.getElementById("interpret-method").value; // corrigé
        await setInterpretationTechnique(technique);
        alert("Méthode définie : " + technique);
    });
    document.getElementById("interpret-btn").addEventListener("click", interpretImage);
    
});
