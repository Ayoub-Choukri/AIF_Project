document.addEventListener("DOMContentLoaded", function () {
    const validateButton = document.getElementById('validate-technique');
    const submitDescriptionButton = document.getElementById('submit-description');
    const embeddingSelect = document.getElementById('embedding-technique');
    const descriptionTextarea = document.getElementById('movie-description');
    const recommendedMoviesArea = document.getElementById('recommended-movies');

    // Fonction pour envoyer le type d'embedding choisi et charger le modèle
    validateButton.addEventListener('click', function () {
        const embeddingType = embeddingSelect.value;

        // Envoi du type d'embedding au backend
        fetch('/Text_Based_Movie_Recommender/set_embedding_type', {
            method: 'POST',
            body: new URLSearchParams({
                'embedding_type': embeddingType
            }),
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded'
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert('Error: ' + data.error);
            } else {
                alert('Embedding technique set successfully!');
                
                // Charger le modèle après avoir validé l'embedding
                loadModel();
            }
        })
        .catch(error => {
            alert('Failed to set embedding technique: ' + error);
        });
    });

    // Fonction pour charger le modèle après avoir défini l'embedding
    function loadModel() {
        fetch('/Text_Based_Movie_Recommender/load_model', {
            method: 'GET'
        })
        .then(response => response.json())
        .then(data => {
            if (data.message) {
                alert('Model loaded successfully!');
            } else {
                alert('Failed to load model: ' + data.error);
            }
        })
        .catch(error => {
            alert('Failed to load model: ' + error);
        });
    }

    // Fonction pour soumettre la description du film et obtenir des recommandations
    submitDescriptionButton.addEventListener('click', function () {
        const movieDescription = descriptionTextarea.value.trim();

        if (!movieDescription) {
            alert('Please enter a movie description.');
            return;
        }

        // Envoi de la description au backend pour obtenir les recommandations
        fetch('/Text_Based_Movie_Recommender/predict', {
            method: 'POST',
            body: new URLSearchParams({
                'movie_description': movieDescription
            }),
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded'
            }
        })
        .then(response => response.json())
        .then(data => {
            // Réinitialiser la zone des recommandations
            recommendedMoviesArea.innerHTML = '';

            // Vérifier les erreurs de la réponse
            if (data.error) {
                alert('Error: ' + data.error);
                return;
            }

            // Insérer les recommandations dans l'interface
            const titles = data.recommended_movies_titles || [];
            const descriptions = data.recommended_movies_descriptions || [];

            titles.forEach((title, index) => {
                const movieElement = document.createElement('div');
                movieElement.classList.add('movie-recommendation');
                movieElement.innerHTML = `
                    <h3 class="movie-title">${title}</h3>
                    <p class="movie-description">${descriptions[index] || 'No description available.'}</p>
                `;
                recommendedMoviesArea.appendChild(movieElement);
            });
        })
        .catch(error => {
            alert('Failed to get recommendations: ' + error);
        });
    });
});
