<?php error_reporting(0);
    // CHECK LOGGED IN OR NOT
session_start();
if(!isset($_SESSION['user_name']) && !isset($_SESSION['name']) )
{
    header("location:login.php"); // REDIRECT TO LOGIN PAGE
}

?>
<!DOCTYPE html>
<html>

<head>
    <title>Welcome,</title>
    <link rel="stylesheet" type="text/css" href="css/homestyle.css">
</head>

<body class="bgimg">

<div class="bgimg">
  <div class="topright">
    <p><b><?php echo "logged-in as : ".$_SESSION['user_name']; ?></a> | <a href="logout.php">Logout</a></p>
  </div>
  <div class="middle">
  <hr>
    <h1><?php echo "Welcome <br/>".$_SESSION['name'];?></h1>
    <p>A Mini Project

By

Joseph, Jewel, Neha and Treza</p>
    <hr>
    <p></p>
  </div>
  <div class="bottomright">
    <p>copyright 2024 all rights reserved</p>
  </div>
</div>

</body>

</html>