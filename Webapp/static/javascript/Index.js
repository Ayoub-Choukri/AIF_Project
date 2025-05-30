// // static/js/script.js

// document.addEventListener("DOMContentLoaded", function () {
//     document.getElementById("redirect-btn").addEventListener("click", function(event) {
//         event.preventDefault();

//         // Récupérer la valeur de l'option sélectionnée
//         var selectedService = document.getElementById("services").value;

//         if (selectedService) {
//             // Envoyer une requête POST pour définir le service
//             fetch('/define_service', {
//                 method: 'POST',
//                 headers: {
//                     'Content-Type': 'application/json'
//                 },
//                 body: JSON.stringify({ Service: selectedService })
//             })
//             .then(response => response.json())
//             .then(data => {
//                 console.log("Service défini avec succès:", data);
//                 // Rediriger vers la page de prédiction
//                 window.location.href = "/prediction_page";
//             })
//             .catch(error => {
//                 console.error("Erreur lors de la définition du service:", error);
//                 alert("Une erreur est survenue. Veuillez réessayer.");
//             });
//         } else {
//             alert("Veuillez sélectionner un service d'abord.");
//         }
//     });
// });



document.addEventListener("DOMContentLoaded", function () {
    document.getElementById("redirect-btn").addEventListener("click", function(event) {
        event.preventDefault();

        // Récupérer la valeur de l'option sélectionnée
        var selectedService = document.getElementById("services").value;

        if (selectedService) {
            // Enregistrer dans le localStorage
            localStorage.setItem("selectedService", selectedService);

            // Rediriger vers l'URL du service sélectionné
            window.location.href = "/" + selectedService;
        } else {
            alert("Veuillez sélectionner un service d'abord.");
        }
    });
});
