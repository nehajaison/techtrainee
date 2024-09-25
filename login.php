<!DOCTYPE html>
<html>

<head>
    <title>LOGIN</title>
    <link rel="stylesheet" type="text/css" href="css/style.css">
</head>

<body>
    <form action="login_check.php" method="post">
        <img src="images/logon.jpg">
        <h2>Welcome Back !</h2>
        <p>Enter your Credentials to access your account</p>
        <?php if (isset($_GET['error'])) { ?> <p class="error"><?php echo $_GET['error']; ?></p> <?php } ?>
        <?php if (isset($_GET['success'])) { ?> <p class="success"><?php echo $_GET['success']; ?></p> <?php } ?>
        <label>Email address</label>
        <input type="email" name="uname" placeholder="Enter Your Email" required>
       
        <label>Password</label>
        <input type="password" name="password" placeholder="Enter Your Password" required>
        <button type="submit">Login</button>
        <p class="links">Or<br/>

<a href="signup.php">Signup</a>
    </form>
</body>

</html>