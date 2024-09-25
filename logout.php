<?php 
// CLEARING ALL SESSION VARIABLES
session_start();
session_unset();
session_destroy();
header("Location: login.php");
?>