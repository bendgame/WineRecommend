function init() {

    // update dropdown menu
    // Grab a reference to the dropdown select element
    var selector = d3.select("#selDataset");

    // Use the list of years  to populate the select options
        d3.json("/country").then((country) => {
        country.forEach((country) => {
            console.log(country);
            selector
            .append("option")
            .text(country.country)
            .property("value", country.country);
            });
            console.log("logging years0");
            console.log(country[0].country);
            const firstYear = country[0].country;
            //run functions on initial page load
            
        });
    // default text before rendering the charts
    var Text1Select = d3.select("#Chart1Text");
    Text1Select
    .append("h3").text("");

}; //end init function

//call function initiate
init();

// ALL -- Used to call functions when a new year is selected in the droplist. 
function optionChanged(year) {
        console.log(year)
        
        //remove map element from html    
        // map.remove()       
        
        // //Run functions on year change
        // createMarkers(year);
        // updateText(year);

    
    };  //end of change function