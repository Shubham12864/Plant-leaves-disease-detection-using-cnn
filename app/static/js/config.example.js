/* Optional: point the frontend at a running inference service.
   Copy this file to js/config.js and load it BEFORE js/api.js on any page:

       <script src="js/config.js"></script>
       <script src="js/api.js"></script>

   Then POST /api/predict and GET /api/model on that base URL are used instead
   of the recorded demo payloads. Nothing else in the pages changes, because
   both paths produce the same view model (see js/scan.js). */

window.LEAFREAD_API = 'http://127.0.0.1:8000';
