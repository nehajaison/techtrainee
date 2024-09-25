<?php 
error_reporting(0);
session_start();
include "db.php";  // DB CONNECTION STRING
if (isset($_POST['name']) && isset($_POST['password']) && isset($_POST['email'])) {  // POST REQUEST CHECK FROM LOGIN FORM


    // VALIDATION FUNCTION TO REMOVE SPECIAL CHARS / TAGS
    function validate($data)
    {
        $data = trim($data);
        $data = stripslashes($data);
        $data = htmlspecialchars($data);
        return $data;
    }



    $name = validate($_POST['name']);
    $email = validate($_POST['email']);
    $pass = validate($_POST['password']);


    if (empty($name)) {  // NO NAME
        header("Location: signup.php?error=Name is required");
        exit();
    }
    else if (empty($email)) {  // NO EMAIL
        header("Location: signup.php?error=Username is required");
        exit();
    } else if (empty($pass)) {  // NO PASSWORD
        header("Location: signup.php?error=Password is required");
        exit();
    } else {
        
        $sql = "SELECT * FROM user WHERE user_name='$email'";
        $result = mysqli_query($conn, $sql);
        if (mysqli_num_rows($result) === 1) {
            header("Location: signup.php?error=This email is already Registered. Please use some other email");
        }else{
        $sql = "INSERT INTO `user` ( `user_name`, `password`, `name`) VALUES ('".$email."', '".$pass."', '".$name."');";
        $result = mysqli_query($conn, $sql);
        header("Location: login.php?success=Successfully Registered. Please login");
        }
    }
} else {

    header("Location: signup.php");
    exit();
}
