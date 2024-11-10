<?php

// DATABASE CONNECTION STRING

$hostname= "localhost";
$username= "root";
$password = "";
$db_name = "project_login";
$conn = mysqli_connect($hostname, $username, $password, $db_name);
if (!$conn) {    echo "Connection failed!";}
?>