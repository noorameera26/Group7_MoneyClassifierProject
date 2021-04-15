package main;

import java.util.Scanner;

public class UserMenu {
    public static int menu() {
        int selection;
        Scanner input = new Scanner(System.in);

        System.out.println("\n--------------------------------------------------------------------------------------------------------------------------------");
        System.out.println("\t\t\t\t\t\t\t\t\t\t\t\t\t\t USER MENU");
        System.out.println("--------------------------------------------------------------------------------------------------------------------------------");
        System.out.println("Choose from these choices:");
        System.out.println("1 - Print the evaluation metrics for testing data set only.");
        System.out.println("2 - Perform the prediction one by one for testing data set and print the evaluation metrics.");
        System.out.println("3 - Quit.");

        System.out.print("\nYour selection: ");
        selection = input.nextInt();
        return selection;
    }
}

