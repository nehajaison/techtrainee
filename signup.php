<!DOCTYPE html>
<html>

<head>
    <title>Signup</title>
    <link rel="stylesheet" type="text/css" href="css/style.css">
</head>

<body>
    <form action="register_user.php" method="post">
        <img src="images/logon.jpg">
        <h2 class="signupheading">Get Started Now</h2>
        <?php if (isset($_GET['error'])) { ?> <p class="error"><?php echo $_GET['error']; ?></p> <?php } ?>
        <label>Your Name</label>
        <input type="test" name="name" placeholder="Enter Your Name" required>
       
        <label>Email address</label>
        <input type="email" name="email" placeholder="Enter Your Email" required>
       
        <label>Password</label>
        <input type="password" name="password" placeholder="Enter Your Password" required>

        
       <p class="tos"><input type="checkbox" name="checkbox" class="checkboxs" required>  &nbsp;&nbsp; I agree to the terms and policy </p>
       
        <button type="submit">Signup</button>
        <p class="links">Or<br/>

        <a href="login.php">Login</a>
        </p>
    </form>
</body>

</html>