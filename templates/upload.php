<!DOCTYPE html>
<html>
<head>
	<title>Uploads Page</title>
</head>
<body>

<?php

$file = $_FILES["file"];
move_uploaded_file($file["tmp_name"], "uploads/" . $file["name"]);
header("Location: userstudy.html");

?>

</body>
</html>

