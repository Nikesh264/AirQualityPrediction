document.getElementById("pollutant-form").addEventListener("submit", function(event) {
	const pm25 = parseFloat(document.querySelector('input[name="pm25"]').value);
	const pm10 = parseFloat(document.querySelector('input[name="pm10"]').value);
	const no2 = parseFloat(document.querySelector('input[name="no2"]').value);
	const so2 = parseFloat(document.querySelector('input[name="so2"]').value);
	const co = parseFloat(document.querySelector('input[name="co"]').value);
	const o3 = parseFloat(document.querySelector('input[name="o3"]').value);

	let errorMessage = "";

	if (pm25 < 0 || pm10 < 0 || no2 < 0 || so2 < 0 || co < 0 || o3 < 0) {
	    errorMessage += "Pollutant levels cannot be negative.\n";
	    event.preventDefault(); // Prevent form submission if there is an error.
	    alert(errorMessage);
	    return;
	  }
});
