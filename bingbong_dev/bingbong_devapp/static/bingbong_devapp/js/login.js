const loginForm = document.getElementById('loginForm');

loginForm.addEventListener('submit', (event) => {
  event.preventDefault();

  const username = document.getElementById('username').value;
  const password = document.getElementById('password').value;

  // Here you'd typically add code to validate the credentials
  // and send them to your backend for authentication.
  // For this simple example, we'll just log them to the console.
  
  console.log('Username:', username);
  console.log('Password:', password);
});
