<?php 
error_reporting(0);
session_start();
include "db.php";  // DB CONNECTION STRING
if (isset($_POST['uname']) && isset($_POST['password'])) {  // POST REQUEST CHECK FROM LOGIN FORM


    // VALIDATION FUNCTION TO REMOVE SPECIAL CHARS / TAGS
    function validate($data)
    {
        $data = trim($data);
        $data = stripslashes($data);
        $data = htmlspecialchars($data);
        return $data;
    }



    $uname = validate($_POST['uname']);
    $pass = validate($_POST['password']);


    if (empty($uname)) {  // NO USERNAME
        header("Location: login.php?error=User Name is required");
        exit();
    } else if (empty($pass)) {  // NO PASSWORD
        header("Location: login.php?error=Password is required");
        exit();
    } else {

        $sql = "SELECT * FROM user WHERE user_name='$uname' AND password='$pass'";
        $result = mysqli_query($conn, $sql);
        if (mysqli_num_rows($result) === 1) {
            $row = mysqli_fetch_assoc($result);
            if ($row['user_name'] === $uname && $row['password'] === $pass) {

                // SETTING SESSION VARIABLES ON SUCCESSFULL LOGIN
                $_SESSION['user_name'] = $row['user_name'];
                $_SESSION['name'] = $row['name'];
                $_SESSION['id'] = $row['id'];

                header("Location:index.php");   // REDIRECT TO HOME PAGE ON SUCCESSFULL LOGIN

            } else {
                header("Location: login.php?error=Incorect User name or password");  // WRONG PASS / USENAME - WRONG PASSWORD
                exit();
            }
        } else {
            header("Location: login.php?error=Incorect User name or password"); // WRONG PASS / USENAME - NO DATA FROM DB
            exit();
        }
    }
} else {

    header("Location: login.php");
    exit();
}
