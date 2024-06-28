let password = document.getElementById ("password");
let confirmpassword = document.getElementById ("confirmpassword");
let showpassword = document.getElementById ("showpassword");

showpassword.onclick = function (){
    if(password.type == "password") {
        password.type = "text";
        confirmpassword.type = "text"
    }else{
        password.type = "password";
        confirmpassword.type = "password"
    }

    password.classList.toggle("show-password");
    confirmpassword.classList.toggle("show-password");
};

function validateForm() {
    // Get form values
    var username = document.getElementById("username").value;
    var email = document.getElementById("email").value;
    var password = document.getElementById("password").value;
    var confirmPassword = document.getElementById("confirmpassword").value;

    if (username.length < 3 || username.length > 30) {
        alert("Username must be between 3 and 30 characters.");
        return false;
    }
    if (!validateEmail(email)) {
        alert("Please enter a valid email address.");
        return false;
    }

    if (password.length < 8) {
        alert("Password must be at least 8 characters.");
        return false;
    }

    if (password !== confirmPassword) {
        alert("Passwords do not match.");
        return false;
    }

    return true; 
}
function validateEmail(email) {
    var re = /\S+@\S+\.\S+/; 
    return re.test(email);
}